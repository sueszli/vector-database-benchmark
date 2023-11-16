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
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.oslogin_v1.common.types import common
from google.cloud.oslogin_v1.services.os_login_service import OsLoginServiceAsyncClient, OsLoginServiceClient, transports
from google.cloud.oslogin_v1.types import oslogin

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
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
    assert OsLoginServiceClient._get_default_mtls_endpoint(None) is None
    assert OsLoginServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert OsLoginServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert OsLoginServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert OsLoginServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert OsLoginServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(OsLoginServiceClient, 'grpc'), (OsLoginServiceAsyncClient, 'grpc_asyncio'), (OsLoginServiceClient, 'rest')])
def test_os_login_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('oslogin.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://oslogin.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.OsLoginServiceGrpcTransport, 'grpc'), (transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.OsLoginServiceRestTransport, 'rest')])
def test_os_login_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(OsLoginServiceClient, 'grpc'), (OsLoginServiceAsyncClient, 'grpc_asyncio'), (OsLoginServiceClient, 'rest')])
def test_os_login_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('oslogin.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://oslogin.googleapis.com')

def test_os_login_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = OsLoginServiceClient.get_transport_class()
    available_transports = [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceRestTransport]
    assert transport in available_transports
    transport = OsLoginServiceClient.get_transport_class('grpc')
    assert transport == transports.OsLoginServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(OsLoginServiceClient, transports.OsLoginServiceGrpcTransport, 'grpc'), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (OsLoginServiceClient, transports.OsLoginServiceRestTransport, 'rest')])
@mock.patch.object(OsLoginServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsLoginServiceClient))
@mock.patch.object(OsLoginServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsLoginServiceAsyncClient))
def test_os_login_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(OsLoginServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(OsLoginServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(OsLoginServiceClient, transports.OsLoginServiceGrpcTransport, 'grpc', 'true'), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (OsLoginServiceClient, transports.OsLoginServiceGrpcTransport, 'grpc', 'false'), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (OsLoginServiceClient, transports.OsLoginServiceRestTransport, 'rest', 'true'), (OsLoginServiceClient, transports.OsLoginServiceRestTransport, 'rest', 'false')])
@mock.patch.object(OsLoginServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsLoginServiceClient))
@mock.patch.object(OsLoginServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsLoginServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_os_login_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [OsLoginServiceClient, OsLoginServiceAsyncClient])
@mock.patch.object(OsLoginServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsLoginServiceClient))
@mock.patch.object(OsLoginServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsLoginServiceAsyncClient))
def test_os_login_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(OsLoginServiceClient, transports.OsLoginServiceGrpcTransport, 'grpc'), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (OsLoginServiceClient, transports.OsLoginServiceRestTransport, 'rest')])
def test_os_login_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(OsLoginServiceClient, transports.OsLoginServiceGrpcTransport, 'grpc', grpc_helpers), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (OsLoginServiceClient, transports.OsLoginServiceRestTransport, 'rest', None)])
def test_os_login_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_os_login_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.oslogin_v1.services.os_login_service.transports.OsLoginServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = OsLoginServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(OsLoginServiceClient, transports.OsLoginServiceGrpcTransport, 'grpc', grpc_helpers), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_os_login_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
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
        create_channel.assert_called_with('oslogin.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly'), scopes=None, default_host='oslogin.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [oslogin.CreateSshPublicKeyRequest, dict])
def test_create_ssh_public_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value')
        response = client.create_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.CreateSshPublicKeyRequest()
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

def test_create_ssh_public_key_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        client.create_ssh_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.CreateSshPublicKeyRequest()

@pytest.mark.asyncio
async def test_create_ssh_public_key_async(transport: str='grpc_asyncio', request_type=oslogin.CreateSshPublicKeyRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value'))
        response = await client.create_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.CreateSshPublicKeyRequest()
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_ssh_public_key_async_from_dict():
    await test_create_ssh_public_key_async(request_type=dict)

def test_create_ssh_public_key_field_headers():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.CreateSshPublicKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        client.create_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_ssh_public_key_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.CreateSshPublicKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey())
        await client.create_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_ssh_public_key_flattened():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        client.create_ssh_public_key(parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ssh_public_key
        mock_val = common.SshPublicKey(key='key_value')
        assert arg == mock_val

def test_create_ssh_public_key_flattened_error():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_ssh_public_key(oslogin.CreateSshPublicKeyRequest(), parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'))

@pytest.mark.asyncio
async def test_create_ssh_public_key_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey())
        response = await client.create_ssh_public_key(parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ssh_public_key
        mock_val = common.SshPublicKey(key='key_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_ssh_public_key_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_ssh_public_key(oslogin.CreateSshPublicKeyRequest(), parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'))

@pytest.mark.parametrize('request_type', [oslogin.DeletePosixAccountRequest, dict])
def test_delete_posix_account(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        call.return_value = None
        response = client.delete_posix_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.DeletePosixAccountRequest()
    assert response is None

def test_delete_posix_account_empty_call():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        client.delete_posix_account()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.DeletePosixAccountRequest()

@pytest.mark.asyncio
async def test_delete_posix_account_async(transport: str='grpc_asyncio', request_type=oslogin.DeletePosixAccountRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_posix_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.DeletePosixAccountRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_posix_account_async_from_dict():
    await test_delete_posix_account_async(request_type=dict)

def test_delete_posix_account_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.DeletePosixAccountRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        call.return_value = None
        client.delete_posix_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_posix_account_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.DeletePosixAccountRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_posix_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_posix_account_flattened():
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        call.return_value = None
        client.delete_posix_account(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_posix_account_flattened_error():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_posix_account(oslogin.DeletePosixAccountRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_posix_account_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_posix_account), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_posix_account(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_posix_account_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_posix_account(oslogin.DeletePosixAccountRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [oslogin.DeleteSshPublicKeyRequest, dict])
def test_delete_ssh_public_key(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        call.return_value = None
        response = client.delete_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.DeleteSshPublicKeyRequest()
    assert response is None

def test_delete_ssh_public_key_empty_call():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        client.delete_ssh_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.DeleteSshPublicKeyRequest()

@pytest.mark.asyncio
async def test_delete_ssh_public_key_async(transport: str='grpc_asyncio', request_type=oslogin.DeleteSshPublicKeyRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.DeleteSshPublicKeyRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_ssh_public_key_async_from_dict():
    await test_delete_ssh_public_key_async(request_type=dict)

def test_delete_ssh_public_key_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.DeleteSshPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        call.return_value = None
        client.delete_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_ssh_public_key_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.DeleteSshPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_ssh_public_key_flattened():
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        call.return_value = None
        client.delete_ssh_public_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_ssh_public_key_flattened_error():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_ssh_public_key(oslogin.DeleteSshPublicKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_ssh_public_key_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_ssh_public_key), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_ssh_public_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_ssh_public_key_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_ssh_public_key(oslogin.DeleteSshPublicKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [oslogin.GetLoginProfileRequest, dict])
def test_get_login_profile(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        call.return_value = oslogin.LoginProfile(name='name_value')
        response = client.get_login_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.GetLoginProfileRequest()
    assert isinstance(response, oslogin.LoginProfile)
    assert response.name == 'name_value'

def test_get_login_profile_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        client.get_login_profile()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.GetLoginProfileRequest()

@pytest.mark.asyncio
async def test_get_login_profile_async(transport: str='grpc_asyncio', request_type=oslogin.GetLoginProfileRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(oslogin.LoginProfile(name='name_value'))
        response = await client.get_login_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.GetLoginProfileRequest()
    assert isinstance(response, oslogin.LoginProfile)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_login_profile_async_from_dict():
    await test_get_login_profile_async(request_type=dict)

def test_get_login_profile_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.GetLoginProfileRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        call.return_value = oslogin.LoginProfile()
        client.get_login_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_login_profile_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.GetLoginProfileRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(oslogin.LoginProfile())
        await client.get_login_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_login_profile_flattened():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        call.return_value = oslogin.LoginProfile()
        client.get_login_profile(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_login_profile_flattened_error():
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_login_profile(oslogin.GetLoginProfileRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_login_profile_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_login_profile), '__call__') as call:
        call.return_value = oslogin.LoginProfile()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(oslogin.LoginProfile())
        response = await client.get_login_profile(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_login_profile_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_login_profile(oslogin.GetLoginProfileRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [oslogin.GetSshPublicKeyRequest, dict])
def test_get_ssh_public_key(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value')
        response = client.get_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.GetSshPublicKeyRequest()
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

def test_get_ssh_public_key_empty_call():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        client.get_ssh_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.GetSshPublicKeyRequest()

@pytest.mark.asyncio
async def test_get_ssh_public_key_async(transport: str='grpc_asyncio', request_type=oslogin.GetSshPublicKeyRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value'))
        response = await client.get_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.GetSshPublicKeyRequest()
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_ssh_public_key_async_from_dict():
    await test_get_ssh_public_key_async(request_type=dict)

def test_get_ssh_public_key_field_headers():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.GetSshPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        client.get_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_ssh_public_key_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.GetSshPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey())
        await client.get_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_ssh_public_key_flattened():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        client.get_ssh_public_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_ssh_public_key_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_ssh_public_key(oslogin.GetSshPublicKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_ssh_public_key_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey())
        response = await client.get_ssh_public_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_ssh_public_key_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_ssh_public_key(oslogin.GetSshPublicKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [oslogin.ImportSshPublicKeyRequest, dict])
def test_import_ssh_public_key(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        call.return_value = oslogin.ImportSshPublicKeyResponse(details='details_value')
        response = client.import_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.ImportSshPublicKeyRequest()
    assert isinstance(response, oslogin.ImportSshPublicKeyResponse)
    assert response.details == 'details_value'

def test_import_ssh_public_key_empty_call():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        client.import_ssh_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.ImportSshPublicKeyRequest()

@pytest.mark.asyncio
async def test_import_ssh_public_key_async(transport: str='grpc_asyncio', request_type=oslogin.ImportSshPublicKeyRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(oslogin.ImportSshPublicKeyResponse(details='details_value'))
        response = await client.import_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.ImportSshPublicKeyRequest()
    assert isinstance(response, oslogin.ImportSshPublicKeyResponse)
    assert response.details == 'details_value'

@pytest.mark.asyncio
async def test_import_ssh_public_key_async_from_dict():
    await test_import_ssh_public_key_async(request_type=dict)

def test_import_ssh_public_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.ImportSshPublicKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        call.return_value = oslogin.ImportSshPublicKeyResponse()
        client.import_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_ssh_public_key_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.ImportSshPublicKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(oslogin.ImportSshPublicKeyResponse())
        await client.import_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_import_ssh_public_key_flattened():
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        call.return_value = oslogin.ImportSshPublicKeyResponse()
        client.import_ssh_public_key(parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'), project_id='project_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ssh_public_key
        mock_val = common.SshPublicKey(key='key_value')
        assert arg == mock_val
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val

def test_import_ssh_public_key_flattened_error():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.import_ssh_public_key(oslogin.ImportSshPublicKeyRequest(), parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'), project_id='project_id_value')

@pytest.mark.asyncio
async def test_import_ssh_public_key_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_ssh_public_key), '__call__') as call:
        call.return_value = oslogin.ImportSshPublicKeyResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(oslogin.ImportSshPublicKeyResponse())
        response = await client.import_ssh_public_key(parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'), project_id='project_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ssh_public_key
        mock_val = common.SshPublicKey(key='key_value')
        assert arg == mock_val
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_import_ssh_public_key_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.import_ssh_public_key(oslogin.ImportSshPublicKeyRequest(), parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'), project_id='project_id_value')

@pytest.mark.parametrize('request_type', [oslogin.UpdateSshPublicKeyRequest, dict])
def test_update_ssh_public_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value')
        response = client.update_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.UpdateSshPublicKeyRequest()
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

def test_update_ssh_public_key_empty_call():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        client.update_ssh_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.UpdateSshPublicKeyRequest()

@pytest.mark.asyncio
async def test_update_ssh_public_key_async(transport: str='grpc_asyncio', request_type=oslogin.UpdateSshPublicKeyRequest):
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value'))
        response = await client.update_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == oslogin.UpdateSshPublicKeyRequest()
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_ssh_public_key_async_from_dict():
    await test_update_ssh_public_key_async(request_type=dict)

def test_update_ssh_public_key_field_headers():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.UpdateSshPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        client.update_ssh_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_ssh_public_key_field_headers_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = oslogin.UpdateSshPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey())
        await client.update_ssh_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_ssh_public_key_flattened():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        client.update_ssh_public_key(name='name_value', ssh_public_key=common.SshPublicKey(key='key_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].ssh_public_key
        mock_val = common.SshPublicKey(key='key_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_ssh_public_key_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_ssh_public_key(oslogin.UpdateSshPublicKeyRequest(), name='name_value', ssh_public_key=common.SshPublicKey(key='key_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_ssh_public_key_flattened_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_ssh_public_key), '__call__') as call:
        call.return_value = common.SshPublicKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(common.SshPublicKey())
        response = await client.update_ssh_public_key(name='name_value', ssh_public_key=common.SshPublicKey(key='key_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].ssh_public_key
        mock_val = common.SshPublicKey(key='key_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_ssh_public_key_flattened_error_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_ssh_public_key(oslogin.UpdateSshPublicKeyRequest(), name='name_value', ssh_public_key=common.SshPublicKey(key='key_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [oslogin.CreateSshPublicKeyRequest, dict])
def test_create_ssh_public_key_rest(request_type):
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'users/sample1'}
    request_init['ssh_public_key'] = {'key': 'key_value', 'expiration_time_usec': 2144, 'fingerprint': 'fingerprint_value', 'name': 'name_value'}
    test_field = oslogin.CreateSshPublicKeyRequest.meta.fields['ssh_public_key']

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
    for (field, value) in request_init['ssh_public_key'].items():
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
                for i in range(0, len(request_init['ssh_public_key'][field])):
                    del request_init['ssh_public_key'][field][i][subfield]
            else:
                del request_init['ssh_public_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = common.SshPublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_ssh_public_key(request)
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

def test_create_ssh_public_key_rest_required_fields(request_type=oslogin.CreateSshPublicKeyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = common.SshPublicKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = common.SshPublicKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_ssh_public_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_ssh_public_key_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_ssh_public_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'sshPublicKey'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_ssh_public_key_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'post_create_ssh_public_key') as post, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_create_ssh_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = oslogin.CreateSshPublicKeyRequest.pb(oslogin.CreateSshPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = common.SshPublicKey.to_json(common.SshPublicKey())
        request = oslogin.CreateSshPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = common.SshPublicKey()
        client.create_ssh_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_ssh_public_key_rest_bad_request(transport: str='rest', request_type=oslogin.CreateSshPublicKeyRequest):
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'users/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_ssh_public_key(request)

def test_create_ssh_public_key_rest_flattened():
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = common.SshPublicKey()
        sample_request = {'parent': 'users/sample1'}
        mock_args = dict(parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = common.SshPublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_ssh_public_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=users/*}/sshPublicKeys' % client.transport._host, args[1])

def test_create_ssh_public_key_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_ssh_public_key(oslogin.CreateSshPublicKeyRequest(), parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'))

def test_create_ssh_public_key_rest_error():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [oslogin.DeletePosixAccountRequest, dict])
def test_delete_posix_account_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/projects/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_posix_account(request)
    assert response is None

def test_delete_posix_account_rest_required_fields(request_type=oslogin.DeletePosixAccountRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_posix_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_posix_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_posix_account(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_posix_account_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_posix_account._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_posix_account_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_delete_posix_account') as pre:
        pre.assert_not_called()
        pb_message = oslogin.DeletePosixAccountRequest.pb(oslogin.DeletePosixAccountRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = oslogin.DeletePosixAccountRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_posix_account(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_posix_account_rest_bad_request(transport: str='rest', request_type=oslogin.DeletePosixAccountRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/projects/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_posix_account(request)

def test_delete_posix_account_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'users/sample1/projects/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_posix_account(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=users/*/projects/*}' % client.transport._host, args[1])

def test_delete_posix_account_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_posix_account(oslogin.DeletePosixAccountRequest(), name='name_value')

def test_delete_posix_account_rest_error():
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [oslogin.DeleteSshPublicKeyRequest, dict])
def test_delete_ssh_public_key_rest(request_type):
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/sshPublicKeys/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_ssh_public_key(request)
    assert response is None

def test_delete_ssh_public_key_rest_required_fields(request_type=oslogin.DeleteSshPublicKeyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_ssh_public_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_ssh_public_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_ssh_public_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_ssh_public_key_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_delete_ssh_public_key') as pre:
        pre.assert_not_called()
        pb_message = oslogin.DeleteSshPublicKeyRequest.pb(oslogin.DeleteSshPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = oslogin.DeleteSshPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_ssh_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_ssh_public_key_rest_bad_request(transport: str='rest', request_type=oslogin.DeleteSshPublicKeyRequest):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/sshPublicKeys/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_ssh_public_key(request)

def test_delete_ssh_public_key_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'users/sample1/sshPublicKeys/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_ssh_public_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=users/*/sshPublicKeys/*}' % client.transport._host, args[1])

def test_delete_ssh_public_key_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_ssh_public_key(oslogin.DeleteSshPublicKeyRequest(), name='name_value')

def test_delete_ssh_public_key_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [oslogin.GetLoginProfileRequest, dict])
def test_get_login_profile_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = oslogin.LoginProfile(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = oslogin.LoginProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_login_profile(request)
    assert isinstance(response, oslogin.LoginProfile)
    assert response.name == 'name_value'

def test_get_login_profile_rest_required_fields(request_type=oslogin.GetLoginProfileRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_login_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_login_profile._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('project_id', 'system_id'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = oslogin.LoginProfile()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = oslogin.LoginProfile.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_login_profile(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_login_profile_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_login_profile._get_unset_required_fields({})
    assert set(unset_fields) == set(('projectId', 'systemId')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_login_profile_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'post_get_login_profile') as post, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_get_login_profile') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = oslogin.GetLoginProfileRequest.pb(oslogin.GetLoginProfileRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = oslogin.LoginProfile.to_json(oslogin.LoginProfile())
        request = oslogin.GetLoginProfileRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = oslogin.LoginProfile()
        client.get_login_profile(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_login_profile_rest_bad_request(transport: str='rest', request_type=oslogin.GetLoginProfileRequest):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_login_profile(request)

def test_get_login_profile_rest_flattened():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = oslogin.LoginProfile()
        sample_request = {'name': 'users/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = oslogin.LoginProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_login_profile(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=users/*}/loginProfile' % client.transport._host, args[1])

def test_get_login_profile_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_login_profile(oslogin.GetLoginProfileRequest(), name='name_value')

def test_get_login_profile_rest_error():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [oslogin.GetSshPublicKeyRequest, dict])
def test_get_ssh_public_key_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/sshPublicKeys/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = common.SshPublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_ssh_public_key(request)
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

def test_get_ssh_public_key_rest_required_fields(request_type=oslogin.GetSshPublicKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = common.SshPublicKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = common.SshPublicKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_ssh_public_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_ssh_public_key_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_ssh_public_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_ssh_public_key_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'post_get_ssh_public_key') as post, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_get_ssh_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = oslogin.GetSshPublicKeyRequest.pb(oslogin.GetSshPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = common.SshPublicKey.to_json(common.SshPublicKey())
        request = oslogin.GetSshPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = common.SshPublicKey()
        client.get_ssh_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_ssh_public_key_rest_bad_request(transport: str='rest', request_type=oslogin.GetSshPublicKeyRequest):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/sshPublicKeys/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_ssh_public_key(request)

def test_get_ssh_public_key_rest_flattened():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = common.SshPublicKey()
        sample_request = {'name': 'users/sample1/sshPublicKeys/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = common.SshPublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_ssh_public_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=users/*/sshPublicKeys/*}' % client.transport._host, args[1])

def test_get_ssh_public_key_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_ssh_public_key(oslogin.GetSshPublicKeyRequest(), name='name_value')

def test_get_ssh_public_key_rest_error():
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [oslogin.ImportSshPublicKeyRequest, dict])
def test_import_ssh_public_key_rest(request_type):
    if False:
        return 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'users/sample1'}
    request_init['ssh_public_key'] = {'key': 'key_value', 'expiration_time_usec': 2144, 'fingerprint': 'fingerprint_value', 'name': 'name_value'}
    test_field = oslogin.ImportSshPublicKeyRequest.meta.fields['ssh_public_key']

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
    for (field, value) in request_init['ssh_public_key'].items():
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
                for i in range(0, len(request_init['ssh_public_key'][field])):
                    del request_init['ssh_public_key'][field][i][subfield]
            else:
                del request_init['ssh_public_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = oslogin.ImportSshPublicKeyResponse(details='details_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = oslogin.ImportSshPublicKeyResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_ssh_public_key(request)
    assert isinstance(response, oslogin.ImportSshPublicKeyResponse)
    assert response.details == 'details_value'

def test_import_ssh_public_key_rest_required_fields(request_type=oslogin.ImportSshPublicKeyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_ssh_public_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('project_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = oslogin.ImportSshPublicKeyResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = oslogin.ImportSshPublicKeyResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.import_ssh_public_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_ssh_public_key_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_ssh_public_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('projectId',)) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_ssh_public_key_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'post_import_ssh_public_key') as post, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_import_ssh_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = oslogin.ImportSshPublicKeyRequest.pb(oslogin.ImportSshPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = oslogin.ImportSshPublicKeyResponse.to_json(oslogin.ImportSshPublicKeyResponse())
        request = oslogin.ImportSshPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = oslogin.ImportSshPublicKeyResponse()
        client.import_ssh_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_ssh_public_key_rest_bad_request(transport: str='rest', request_type=oslogin.ImportSshPublicKeyRequest):
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'users/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_ssh_public_key(request)

def test_import_ssh_public_key_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = oslogin.ImportSshPublicKeyResponse()
        sample_request = {'parent': 'users/sample1'}
        mock_args = dict(parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'), project_id='project_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = oslogin.ImportSshPublicKeyResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.import_ssh_public_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=users/*}:importSshPublicKey' % client.transport._host, args[1])

def test_import_ssh_public_key_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.import_ssh_public_key(oslogin.ImportSshPublicKeyRequest(), parent='parent_value', ssh_public_key=common.SshPublicKey(key='key_value'), project_id='project_id_value')

def test_import_ssh_public_key_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [oslogin.UpdateSshPublicKeyRequest, dict])
def test_update_ssh_public_key_rest(request_type):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/sshPublicKeys/sample2'}
    request_init['ssh_public_key'] = {'key': 'key_value', 'expiration_time_usec': 2144, 'fingerprint': 'fingerprint_value', 'name': 'name_value'}
    test_field = oslogin.UpdateSshPublicKeyRequest.meta.fields['ssh_public_key']

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
    for (field, value) in request_init['ssh_public_key'].items():
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
                for i in range(0, len(request_init['ssh_public_key'][field])):
                    del request_init['ssh_public_key'][field][i][subfield]
            else:
                del request_init['ssh_public_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = common.SshPublicKey(key='key_value', expiration_time_usec=2144, fingerprint='fingerprint_value', name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = common.SshPublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_ssh_public_key(request)
    assert isinstance(response, common.SshPublicKey)
    assert response.key == 'key_value'
    assert response.expiration_time_usec == 2144
    assert response.fingerprint == 'fingerprint_value'
    assert response.name == 'name_value'

def test_update_ssh_public_key_rest_required_fields(request_type=oslogin.UpdateSshPublicKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsLoginServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_ssh_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_ssh_public_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = common.SshPublicKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = common.SshPublicKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_ssh_public_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_ssh_public_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_ssh_public_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('name', 'sshPublicKey'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_ssh_public_key_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsLoginServiceRestInterceptor())
    client = OsLoginServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'post_update_ssh_public_key') as post, mock.patch.object(transports.OsLoginServiceRestInterceptor, 'pre_update_ssh_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = oslogin.UpdateSshPublicKeyRequest.pb(oslogin.UpdateSshPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = common.SshPublicKey.to_json(common.SshPublicKey())
        request = oslogin.UpdateSshPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = common.SshPublicKey()
        client.update_ssh_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_ssh_public_key_rest_bad_request(transport: str='rest', request_type=oslogin.UpdateSshPublicKeyRequest):
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/sshPublicKeys/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_ssh_public_key(request)

def test_update_ssh_public_key_rest_flattened():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = common.SshPublicKey()
        sample_request = {'name': 'users/sample1/sshPublicKeys/sample2'}
        mock_args = dict(name='name_value', ssh_public_key=common.SshPublicKey(key='key_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = common.SshPublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_ssh_public_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=users/*/sshPublicKeys/*}' % client.transport._host, args[1])

def test_update_ssh_public_key_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_ssh_public_key(oslogin.UpdateSshPublicKeyRequest(), name='name_value', ssh_public_key=common.SshPublicKey(key='key_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_ssh_public_key_rest_error():
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.OsLoginServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.OsLoginServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsLoginServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.OsLoginServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = OsLoginServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = OsLoginServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.OsLoginServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsLoginServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsLoginServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = OsLoginServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.OsLoginServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.OsLoginServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceGrpcAsyncIOTransport, transports.OsLoginServiceRestTransport])
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
    transport = OsLoginServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.OsLoginServiceGrpcTransport)

def test_os_login_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.OsLoginServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_os_login_service_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.oslogin_v1.services.os_login_service.transports.OsLoginServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.OsLoginServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_ssh_public_key', 'delete_posix_account', 'delete_ssh_public_key', 'get_login_profile', 'get_ssh_public_key', 'import_ssh_public_key', 'update_ssh_public_key')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_os_login_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.oslogin_v1.services.os_login_service.transports.OsLoginServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.OsLoginServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly'), quota_project_id='octopus')

def test_os_login_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.oslogin_v1.services.os_login_service.transports.OsLoginServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.OsLoginServiceTransport()
        adc.assert_called_once()

def test_os_login_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        OsLoginServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceGrpcAsyncIOTransport])
def test_os_login_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceGrpcAsyncIOTransport, transports.OsLoginServiceRestTransport])
def test_os_login_service_transport_auth_gdch_credentials(transport_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.OsLoginServiceGrpcTransport, grpc_helpers), (transports.OsLoginServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_os_login_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('oslogin.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly'), scopes=['1', '2'], default_host='oslogin.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceGrpcAsyncIOTransport])
def test_os_login_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_os_login_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.OsLoginServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_os_login_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='oslogin.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('oslogin.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://oslogin.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_os_login_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='oslogin.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('oslogin.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://oslogin.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_os_login_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = OsLoginServiceClient(credentials=creds1, transport=transport_name)
    client2 = OsLoginServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_ssh_public_key._session
    session2 = client2.transport.create_ssh_public_key._session
    assert session1 != session2
    session1 = client1.transport.delete_posix_account._session
    session2 = client2.transport.delete_posix_account._session
    assert session1 != session2
    session1 = client1.transport.delete_ssh_public_key._session
    session2 = client2.transport.delete_ssh_public_key._session
    assert session1 != session2
    session1 = client1.transport.get_login_profile._session
    session2 = client2.transport.get_login_profile._session
    assert session1 != session2
    session1 = client1.transport.get_ssh_public_key._session
    session2 = client2.transport.get_ssh_public_key._session
    assert session1 != session2
    session1 = client1.transport.import_ssh_public_key._session
    session2 = client2.transport.import_ssh_public_key._session
    assert session1 != session2
    session1 = client1.transport.update_ssh_public_key._session
    session2 = client2.transport.update_ssh_public_key._session
    assert session1 != session2

def test_os_login_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.OsLoginServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_os_login_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.OsLoginServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceGrpcAsyncIOTransport])
def test_os_login_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class', [transports.OsLoginServiceGrpcTransport, transports.OsLoginServiceGrpcAsyncIOTransport])
def test_os_login_service_transport_channel_mtls_with_adc(transport_class):
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

def test_posix_account_path():
    if False:
        i = 10
        return i + 15
    user = 'squid'
    project = 'clam'
    expected = 'users/{user}/projects/{project}'.format(user=user, project=project)
    actual = OsLoginServiceClient.posix_account_path(user, project)
    assert expected == actual

def test_parse_posix_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'user': 'whelk', 'project': 'octopus'}
    path = OsLoginServiceClient.posix_account_path(**expected)
    actual = OsLoginServiceClient.parse_posix_account_path(path)
    assert expected == actual

def test_ssh_public_key_path():
    if False:
        for i in range(10):
            print('nop')
    user = 'oyster'
    fingerprint = 'nudibranch'
    expected = 'users/{user}/sshPublicKeys/{fingerprint}'.format(user=user, fingerprint=fingerprint)
    actual = OsLoginServiceClient.ssh_public_key_path(user, fingerprint)
    assert expected == actual

def test_parse_ssh_public_key_path():
    if False:
        print('Hello World!')
    expected = {'user': 'cuttlefish', 'fingerprint': 'mussel'}
    path = OsLoginServiceClient.ssh_public_key_path(**expected)
    actual = OsLoginServiceClient.parse_ssh_public_key_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = OsLoginServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nautilus'}
    path = OsLoginServiceClient.common_billing_account_path(**expected)
    actual = OsLoginServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = OsLoginServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'abalone'}
    path = OsLoginServiceClient.common_folder_path(**expected)
    actual = OsLoginServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = OsLoginServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'clam'}
    path = OsLoginServiceClient.common_organization_path(**expected)
    actual = OsLoginServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = OsLoginServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus'}
    path = OsLoginServiceClient.common_project_path(**expected)
    actual = OsLoginServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = OsLoginServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = OsLoginServiceClient.common_location_path(**expected)
    actual = OsLoginServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.OsLoginServiceTransport, '_prep_wrapped_messages') as prep:
        client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.OsLoginServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = OsLoginServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = OsLoginServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = OsLoginServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(OsLoginServiceClient, transports.OsLoginServiceGrpcTransport), (OsLoginServiceAsyncClient, transports.OsLoginServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)