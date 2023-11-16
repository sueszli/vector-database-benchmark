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
from google.longrunning import operations_pb2
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
from google.ai.generativelanguage_v1beta3.services.permission_service import PermissionServiceAsyncClient, PermissionServiceClient, pagers, transports
from google.ai.generativelanguage_v1beta3.types import permission as gag_permission
from google.ai.generativelanguage_v1beta3.types import permission
from google.ai.generativelanguage_v1beta3.types import permission_service

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
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert PermissionServiceClient._get_default_mtls_endpoint(None) is None
    assert PermissionServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert PermissionServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert PermissionServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert PermissionServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert PermissionServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(PermissionServiceClient, 'grpc'), (PermissionServiceAsyncClient, 'grpc_asyncio'), (PermissionServiceClient, 'rest')])
def test_permission_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('generativelanguage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.PermissionServiceGrpcTransport, 'grpc'), (transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.PermissionServiceRestTransport, 'rest')])
def test_permission_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(PermissionServiceClient, 'grpc'), (PermissionServiceAsyncClient, 'grpc_asyncio'), (PermissionServiceClient, 'rest')])
def test_permission_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('generativelanguage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com')

def test_permission_service_client_get_transport_class():
    if False:
        return 10
    transport = PermissionServiceClient.get_transport_class()
    available_transports = [transports.PermissionServiceGrpcTransport, transports.PermissionServiceRestTransport]
    assert transport in available_transports
    transport = PermissionServiceClient.get_transport_class('grpc')
    assert transport == transports.PermissionServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(PermissionServiceClient, transports.PermissionServiceGrpcTransport, 'grpc'), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (PermissionServiceClient, transports.PermissionServiceRestTransport, 'rest')])
@mock.patch.object(PermissionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PermissionServiceClient))
@mock.patch.object(PermissionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PermissionServiceAsyncClient))
def test_permission_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(PermissionServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(PermissionServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(PermissionServiceClient, transports.PermissionServiceGrpcTransport, 'grpc', 'true'), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (PermissionServiceClient, transports.PermissionServiceGrpcTransport, 'grpc', 'false'), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (PermissionServiceClient, transports.PermissionServiceRestTransport, 'rest', 'true'), (PermissionServiceClient, transports.PermissionServiceRestTransport, 'rest', 'false')])
@mock.patch.object(PermissionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PermissionServiceClient))
@mock.patch.object(PermissionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PermissionServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_permission_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [PermissionServiceClient, PermissionServiceAsyncClient])
@mock.patch.object(PermissionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PermissionServiceClient))
@mock.patch.object(PermissionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PermissionServiceAsyncClient))
def test_permission_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(PermissionServiceClient, transports.PermissionServiceGrpcTransport, 'grpc'), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (PermissionServiceClient, transports.PermissionServiceRestTransport, 'rest')])
def test_permission_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(PermissionServiceClient, transports.PermissionServiceGrpcTransport, 'grpc', grpc_helpers), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (PermissionServiceClient, transports.PermissionServiceRestTransport, 'rest', None)])
def test_permission_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_permission_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.ai.generativelanguage_v1beta3.services.permission_service.transports.PermissionServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = PermissionServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(PermissionServiceClient, transports.PermissionServiceGrpcTransport, 'grpc', grpc_helpers), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_permission_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('generativelanguage.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=(), scopes=None, default_host='generativelanguage.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [permission_service.CreatePermissionRequest, dict])
def test_create_permission(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        call.return_value = gag_permission.Permission(name='name_value', grantee_type=gag_permission.Permission.GranteeType.USER, email_address='email_address_value', role=gag_permission.Permission.Role.OWNER)
        response = client.create_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.CreatePermissionRequest()
    assert isinstance(response, gag_permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == gag_permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == gag_permission.Permission.Role.OWNER

def test_create_permission_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        client.create_permission()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.CreatePermissionRequest()

@pytest.mark.asyncio
async def test_create_permission_async(transport: str='grpc_asyncio', request_type=permission_service.CreatePermissionRequest):
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gag_permission.Permission(name='name_value', grantee_type=gag_permission.Permission.GranteeType.USER, email_address='email_address_value', role=gag_permission.Permission.Role.OWNER))
        response = await client.create_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.CreatePermissionRequest()
    assert isinstance(response, gag_permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == gag_permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == gag_permission.Permission.Role.OWNER

@pytest.mark.asyncio
async def test_create_permission_async_from_dict():
    await test_create_permission_async(request_type=dict)

def test_create_permission_field_headers():
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.CreatePermissionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        call.return_value = gag_permission.Permission()
        client.create_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_permission_field_headers_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.CreatePermissionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gag_permission.Permission())
        await client.create_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_permission_flattened():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        call.return_value = gag_permission.Permission()
        client.create_permission(parent='parent_value', permission=gag_permission.Permission(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].permission
        mock_val = gag_permission.Permission(name='name_value')
        assert arg == mock_val

def test_create_permission_flattened_error():
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_permission(permission_service.CreatePermissionRequest(), parent='parent_value', permission=gag_permission.Permission(name='name_value'))

@pytest.mark.asyncio
async def test_create_permission_flattened_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_permission), '__call__') as call:
        call.return_value = gag_permission.Permission()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gag_permission.Permission())
        response = await client.create_permission(parent='parent_value', permission=gag_permission.Permission(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].permission
        mock_val = gag_permission.Permission(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_permission_flattened_error_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_permission(permission_service.CreatePermissionRequest(), parent='parent_value', permission=gag_permission.Permission(name='name_value'))

@pytest.mark.parametrize('request_type', [permission_service.GetPermissionRequest, dict])
def test_get_permission(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        call.return_value = permission.Permission(name='name_value', grantee_type=permission.Permission.GranteeType.USER, email_address='email_address_value', role=permission.Permission.Role.OWNER)
        response = client.get_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.GetPermissionRequest()
    assert isinstance(response, permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == permission.Permission.Role.OWNER

def test_get_permission_empty_call():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        client.get_permission()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.GetPermissionRequest()

@pytest.mark.asyncio
async def test_get_permission_async(transport: str='grpc_asyncio', request_type=permission_service.GetPermissionRequest):
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission.Permission(name='name_value', grantee_type=permission.Permission.GranteeType.USER, email_address='email_address_value', role=permission.Permission.Role.OWNER))
        response = await client.get_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.GetPermissionRequest()
    assert isinstance(response, permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == permission.Permission.Role.OWNER

@pytest.mark.asyncio
async def test_get_permission_async_from_dict():
    await test_get_permission_async(request_type=dict)

def test_get_permission_field_headers():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.GetPermissionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        call.return_value = permission.Permission()
        client.get_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_permission_field_headers_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.GetPermissionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission.Permission())
        await client.get_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_permission_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        call.return_value = permission.Permission()
        client.get_permission(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_permission_flattened_error():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_permission(permission_service.GetPermissionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_permission_flattened_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_permission), '__call__') as call:
        call.return_value = permission.Permission()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission.Permission())
        response = await client.get_permission(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_permission_flattened_error_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_permission(permission_service.GetPermissionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [permission_service.ListPermissionsRequest, dict])
def test_list_permissions(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.return_value = permission_service.ListPermissionsResponse(next_page_token='next_page_token_value')
        response = client.list_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.ListPermissionsRequest()
    assert isinstance(response, pagers.ListPermissionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_permissions_empty_call():
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        client.list_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.ListPermissionsRequest()

@pytest.mark.asyncio
async def test_list_permissions_async(transport: str='grpc_asyncio', request_type=permission_service.ListPermissionsRequest):
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission_service.ListPermissionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.ListPermissionsRequest()
    assert isinstance(response, pagers.ListPermissionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_permissions_async_from_dict():
    await test_list_permissions_async(request_type=dict)

def test_list_permissions_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.ListPermissionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.return_value = permission_service.ListPermissionsResponse()
        client.list_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_permissions_field_headers_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.ListPermissionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission_service.ListPermissionsResponse())
        await client.list_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_permissions_flattened():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.return_value = permission_service.ListPermissionsResponse()
        client.list_permissions(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_permissions_flattened_error():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_permissions(permission_service.ListPermissionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_permissions_flattened_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.return_value = permission_service.ListPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission_service.ListPermissionsResponse())
        response = await client.list_permissions(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_permissions_flattened_error_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_permissions(permission_service.ListPermissionsRequest(), parent='parent_value')

def test_list_permissions_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.side_effect = (permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission(), permission.Permission()], next_page_token='abc'), permission_service.ListPermissionsResponse(permissions=[], next_page_token='def'), permission_service.ListPermissionsResponse(permissions=[permission.Permission()], next_page_token='ghi'), permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_permissions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, permission.Permission) for i in results))

def test_list_permissions_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_permissions), '__call__') as call:
        call.side_effect = (permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission(), permission.Permission()], next_page_token='abc'), permission_service.ListPermissionsResponse(permissions=[], next_page_token='def'), permission_service.ListPermissionsResponse(permissions=[permission.Permission()], next_page_token='ghi'), permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission()]), RuntimeError)
        pages = list(client.list_permissions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_permissions_async_pager():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_permissions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission(), permission.Permission()], next_page_token='abc'), permission_service.ListPermissionsResponse(permissions=[], next_page_token='def'), permission_service.ListPermissionsResponse(permissions=[permission.Permission()], next_page_token='ghi'), permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission()]), RuntimeError)
        async_pager = await client.list_permissions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, permission.Permission) for i in responses))

@pytest.mark.asyncio
async def test_list_permissions_async_pages():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_permissions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission(), permission.Permission()], next_page_token='abc'), permission_service.ListPermissionsResponse(permissions=[], next_page_token='def'), permission_service.ListPermissionsResponse(permissions=[permission.Permission()], next_page_token='ghi'), permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_permissions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [permission_service.UpdatePermissionRequest, dict])
def test_update_permission(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        call.return_value = gag_permission.Permission(name='name_value', grantee_type=gag_permission.Permission.GranteeType.USER, email_address='email_address_value', role=gag_permission.Permission.Role.OWNER)
        response = client.update_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.UpdatePermissionRequest()
    assert isinstance(response, gag_permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == gag_permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == gag_permission.Permission.Role.OWNER

def test_update_permission_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        client.update_permission()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.UpdatePermissionRequest()

@pytest.mark.asyncio
async def test_update_permission_async(transport: str='grpc_asyncio', request_type=permission_service.UpdatePermissionRequest):
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gag_permission.Permission(name='name_value', grantee_type=gag_permission.Permission.GranteeType.USER, email_address='email_address_value', role=gag_permission.Permission.Role.OWNER))
        response = await client.update_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.UpdatePermissionRequest()
    assert isinstance(response, gag_permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == gag_permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == gag_permission.Permission.Role.OWNER

@pytest.mark.asyncio
async def test_update_permission_async_from_dict():
    await test_update_permission_async(request_type=dict)

def test_update_permission_field_headers():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.UpdatePermissionRequest()
    request.permission.name = 'name_value'
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        call.return_value = gag_permission.Permission()
        client.update_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'permission.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_permission_field_headers_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.UpdatePermissionRequest()
    request.permission.name = 'name_value'
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gag_permission.Permission())
        await client.update_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'permission.name=name_value') in kw['metadata']

def test_update_permission_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        call.return_value = gag_permission.Permission()
        client.update_permission(permission=gag_permission.Permission(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].permission
        mock_val = gag_permission.Permission(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_permission_flattened_error():
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_permission(permission_service.UpdatePermissionRequest(), permission=gag_permission.Permission(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_permission_flattened_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_permission), '__call__') as call:
        call.return_value = gag_permission.Permission()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gag_permission.Permission())
        response = await client.update_permission(permission=gag_permission.Permission(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].permission
        mock_val = gag_permission.Permission(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_permission_flattened_error_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_permission(permission_service.UpdatePermissionRequest(), permission=gag_permission.Permission(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [permission_service.DeletePermissionRequest, dict])
def test_delete_permission(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        call.return_value = None
        response = client.delete_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.DeletePermissionRequest()
    assert response is None

def test_delete_permission_empty_call():
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        client.delete_permission()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.DeletePermissionRequest()

@pytest.mark.asyncio
async def test_delete_permission_async(transport: str='grpc_asyncio', request_type=permission_service.DeletePermissionRequest):
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.DeletePermissionRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_permission_async_from_dict():
    await test_delete_permission_async(request_type=dict)

def test_delete_permission_field_headers():
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.DeletePermissionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        call.return_value = None
        client.delete_permission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_permission_field_headers_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.DeletePermissionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_permission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_permission_flattened():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        call.return_value = None
        client.delete_permission(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_permission_flattened_error():
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_permission(permission_service.DeletePermissionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_permission_flattened_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_permission), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_permission(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_permission_flattened_error_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_permission(permission_service.DeletePermissionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [permission_service.TransferOwnershipRequest, dict])
def test_transfer_ownership(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.transfer_ownership), '__call__') as call:
        call.return_value = permission_service.TransferOwnershipResponse()
        response = client.transfer_ownership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.TransferOwnershipRequest()
    assert isinstance(response, permission_service.TransferOwnershipResponse)

def test_transfer_ownership_empty_call():
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.transfer_ownership), '__call__') as call:
        client.transfer_ownership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.TransferOwnershipRequest()

@pytest.mark.asyncio
async def test_transfer_ownership_async(transport: str='grpc_asyncio', request_type=permission_service.TransferOwnershipRequest):
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.transfer_ownership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission_service.TransferOwnershipResponse())
        response = await client.transfer_ownership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == permission_service.TransferOwnershipRequest()
    assert isinstance(response, permission_service.TransferOwnershipResponse)

@pytest.mark.asyncio
async def test_transfer_ownership_async_from_dict():
    await test_transfer_ownership_async(request_type=dict)

def test_transfer_ownership_field_headers():
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.TransferOwnershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.transfer_ownership), '__call__') as call:
        call.return_value = permission_service.TransferOwnershipResponse()
        client.transfer_ownership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_transfer_ownership_field_headers_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = permission_service.TransferOwnershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.transfer_ownership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(permission_service.TransferOwnershipResponse())
        await client.transfer_ownership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [permission_service.CreatePermissionRequest, dict])
def test_create_permission_rest(request_type):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tunedModels/sample1'}
    request_init['permission'] = {'name': 'name_value', 'grantee_type': 1, 'email_address': 'email_address_value', 'role': 1}
    test_field = permission_service.CreatePermissionRequest.meta.fields['permission']

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
    for (field, value) in request_init['permission'].items():
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
                for i in range(0, len(request_init['permission'][field])):
                    del request_init['permission'][field][i][subfield]
            else:
                del request_init['permission'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gag_permission.Permission(name='name_value', grantee_type=gag_permission.Permission.GranteeType.USER, email_address='email_address_value', role=gag_permission.Permission.Role.OWNER)
        response_value = Response()
        response_value.status_code = 200
        return_value = gag_permission.Permission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_permission(request)
    assert isinstance(response, gag_permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == gag_permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == gag_permission.Permission.Role.OWNER

def test_create_permission_rest_required_fields(request_type=permission_service.CreatePermissionRequest):
    if False:
        return 10
    transport_class = transports.PermissionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gag_permission.Permission()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gag_permission.Permission.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_permission(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_permission_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_permission._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'permission'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_permission_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.PermissionServiceRestInterceptor())
    client = PermissionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.PermissionServiceRestInterceptor, 'post_create_permission') as post, mock.patch.object(transports.PermissionServiceRestInterceptor, 'pre_create_permission') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = permission_service.CreatePermissionRequest.pb(permission_service.CreatePermissionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gag_permission.Permission.to_json(gag_permission.Permission())
        request = permission_service.CreatePermissionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gag_permission.Permission()
        client.create_permission(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_permission_rest_bad_request(transport: str='rest', request_type=permission_service.CreatePermissionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tunedModels/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_permission(request)

def test_create_permission_rest_flattened():
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gag_permission.Permission()
        sample_request = {'parent': 'tunedModels/sample1'}
        mock_args = dict(parent='parent_value', permission=gag_permission.Permission(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gag_permission.Permission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_permission(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{parent=tunedModels/*}/permissions' % client.transport._host, args[1])

def test_create_permission_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_permission(permission_service.CreatePermissionRequest(), parent='parent_value', permission=gag_permission.Permission(name='name_value'))

def test_create_permission_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [permission_service.GetPermissionRequest, dict])
def test_get_permission_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tunedModels/sample1/permissions/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = permission.Permission(name='name_value', grantee_type=permission.Permission.GranteeType.USER, email_address='email_address_value', role=permission.Permission.Role.OWNER)
        response_value = Response()
        response_value.status_code = 200
        return_value = permission.Permission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_permission(request)
    assert isinstance(response, permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == permission.Permission.Role.OWNER

def test_get_permission_rest_required_fields(request_type=permission_service.GetPermissionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.PermissionServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = permission.Permission()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = permission.Permission.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_permission(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_permission_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_permission._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_permission_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.PermissionServiceRestInterceptor())
    client = PermissionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.PermissionServiceRestInterceptor, 'post_get_permission') as post, mock.patch.object(transports.PermissionServiceRestInterceptor, 'pre_get_permission') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = permission_service.GetPermissionRequest.pb(permission_service.GetPermissionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = permission.Permission.to_json(permission.Permission())
        request = permission_service.GetPermissionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = permission.Permission()
        client.get_permission(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_permission_rest_bad_request(transport: str='rest', request_type=permission_service.GetPermissionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tunedModels/sample1/permissions/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_permission(request)

def test_get_permission_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = permission.Permission()
        sample_request = {'name': 'tunedModels/sample1/permissions/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = permission.Permission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_permission(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{name=tunedModels/*/permissions/*}' % client.transport._host, args[1])

def test_get_permission_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_permission(permission_service.GetPermissionRequest(), name='name_value')

def test_get_permission_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [permission_service.ListPermissionsRequest, dict])
def test_list_permissions_rest(request_type):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tunedModels/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = permission_service.ListPermissionsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = permission_service.ListPermissionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_permissions(request)
    assert isinstance(response, pagers.ListPermissionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_permissions_rest_required_fields(request_type=permission_service.ListPermissionsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.PermissionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_permissions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = permission_service.ListPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = permission_service.ListPermissionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_permissions_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_permissions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.PermissionServiceRestInterceptor())
    client = PermissionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.PermissionServiceRestInterceptor, 'post_list_permissions') as post, mock.patch.object(transports.PermissionServiceRestInterceptor, 'pre_list_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = permission_service.ListPermissionsRequest.pb(permission_service.ListPermissionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = permission_service.ListPermissionsResponse.to_json(permission_service.ListPermissionsResponse())
        request = permission_service.ListPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = permission_service.ListPermissionsResponse()
        client.list_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_permissions_rest_bad_request(transport: str='rest', request_type=permission_service.ListPermissionsRequest):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tunedModels/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_permissions(request)

def test_list_permissions_rest_flattened():
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = permission_service.ListPermissionsResponse()
        sample_request = {'parent': 'tunedModels/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = permission_service.ListPermissionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{parent=tunedModels/*}/permissions' % client.transport._host, args[1])

def test_list_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_permissions(permission_service.ListPermissionsRequest(), parent='parent_value')

def test_list_permissions_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission(), permission.Permission()], next_page_token='abc'), permission_service.ListPermissionsResponse(permissions=[], next_page_token='def'), permission_service.ListPermissionsResponse(permissions=[permission.Permission()], next_page_token='ghi'), permission_service.ListPermissionsResponse(permissions=[permission.Permission(), permission.Permission()]))
        response = response + response
        response = tuple((permission_service.ListPermissionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'tunedModels/sample1'}
        pager = client.list_permissions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, permission.Permission) for i in results))
        pages = list(client.list_permissions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [permission_service.UpdatePermissionRequest, dict])
def test_update_permission_rest(request_type):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'permission': {'name': 'tunedModels/sample1/permissions/sample2'}}
    request_init['permission'] = {'name': 'tunedModels/sample1/permissions/sample2', 'grantee_type': 1, 'email_address': 'email_address_value', 'role': 1}
    test_field = permission_service.UpdatePermissionRequest.meta.fields['permission']

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
    for (field, value) in request_init['permission'].items():
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
                for i in range(0, len(request_init['permission'][field])):
                    del request_init['permission'][field][i][subfield]
            else:
                del request_init['permission'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gag_permission.Permission(name='name_value', grantee_type=gag_permission.Permission.GranteeType.USER, email_address='email_address_value', role=gag_permission.Permission.Role.OWNER)
        response_value = Response()
        response_value.status_code = 200
        return_value = gag_permission.Permission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_permission(request)
    assert isinstance(response, gag_permission.Permission)
    assert response.name == 'name_value'
    assert response.grantee_type == gag_permission.Permission.GranteeType.USER
    assert response.email_address == 'email_address_value'
    assert response.role == gag_permission.Permission.Role.OWNER

def test_update_permission_rest_required_fields(request_type=permission_service.UpdatePermissionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.PermissionServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_permission._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gag_permission.Permission()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gag_permission.Permission.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_permission(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_permission_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_permission._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('permission', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_permission_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.PermissionServiceRestInterceptor())
    client = PermissionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.PermissionServiceRestInterceptor, 'post_update_permission') as post, mock.patch.object(transports.PermissionServiceRestInterceptor, 'pre_update_permission') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = permission_service.UpdatePermissionRequest.pb(permission_service.UpdatePermissionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gag_permission.Permission.to_json(gag_permission.Permission())
        request = permission_service.UpdatePermissionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gag_permission.Permission()
        client.update_permission(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_permission_rest_bad_request(transport: str='rest', request_type=permission_service.UpdatePermissionRequest):
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'permission': {'name': 'tunedModels/sample1/permissions/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_permission(request)

def test_update_permission_rest_flattened():
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gag_permission.Permission()
        sample_request = {'permission': {'name': 'tunedModels/sample1/permissions/sample2'}}
        mock_args = dict(permission=gag_permission.Permission(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gag_permission.Permission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_permission(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{permission.name=tunedModels/*/permissions/*}' % client.transport._host, args[1])

def test_update_permission_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_permission(permission_service.UpdatePermissionRequest(), permission=gag_permission.Permission(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_permission_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [permission_service.DeletePermissionRequest, dict])
def test_delete_permission_rest(request_type):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tunedModels/sample1/permissions/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_permission(request)
    assert response is None

def test_delete_permission_rest_required_fields(request_type=permission_service.DeletePermissionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.PermissionServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_permission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_permission(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_permission_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_permission._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_permission_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.PermissionServiceRestInterceptor())
    client = PermissionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.PermissionServiceRestInterceptor, 'pre_delete_permission') as pre:
        pre.assert_not_called()
        pb_message = permission_service.DeletePermissionRequest.pb(permission_service.DeletePermissionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = permission_service.DeletePermissionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_permission(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_permission_rest_bad_request(transport: str='rest', request_type=permission_service.DeletePermissionRequest):
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tunedModels/sample1/permissions/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_permission(request)

def test_delete_permission_rest_flattened():
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'tunedModels/sample1/permissions/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_permission(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{name=tunedModels/*/permissions/*}' % client.transport._host, args[1])

def test_delete_permission_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_permission(permission_service.DeletePermissionRequest(), name='name_value')

def test_delete_permission_rest_error():
    if False:
        i = 10
        return i + 15
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [permission_service.TransferOwnershipRequest, dict])
def test_transfer_ownership_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tunedModels/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = permission_service.TransferOwnershipResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = permission_service.TransferOwnershipResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.transfer_ownership(request)
    assert isinstance(response, permission_service.TransferOwnershipResponse)

def test_transfer_ownership_rest_required_fields(request_type=permission_service.TransferOwnershipRequest):
    if False:
        return 10
    transport_class = transports.PermissionServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['email_address'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).transfer_ownership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['emailAddress'] = 'email_address_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).transfer_ownership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'emailAddress' in jsonified_request
    assert jsonified_request['emailAddress'] == 'email_address_value'
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = permission_service.TransferOwnershipResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = permission_service.TransferOwnershipResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.transfer_ownership(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_transfer_ownership_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.transfer_ownership._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'emailAddress'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_transfer_ownership_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.PermissionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.PermissionServiceRestInterceptor())
    client = PermissionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.PermissionServiceRestInterceptor, 'post_transfer_ownership') as post, mock.patch.object(transports.PermissionServiceRestInterceptor, 'pre_transfer_ownership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = permission_service.TransferOwnershipRequest.pb(permission_service.TransferOwnershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = permission_service.TransferOwnershipResponse.to_json(permission_service.TransferOwnershipResponse())
        request = permission_service.TransferOwnershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = permission_service.TransferOwnershipResponse()
        client.transfer_ownership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_transfer_ownership_rest_bad_request(transport: str='rest', request_type=permission_service.TransferOwnershipRequest):
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tunedModels/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.transfer_ownership(request)

def test_transfer_ownership_rest_error():
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.PermissionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.PermissionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = PermissionServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.PermissionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = PermissionServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = PermissionServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.PermissionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = PermissionServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.PermissionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = PermissionServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.PermissionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.PermissionServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.PermissionServiceGrpcTransport, transports.PermissionServiceGrpcAsyncIOTransport, transports.PermissionServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = PermissionServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.PermissionServiceGrpcTransport)

def test_permission_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.PermissionServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_permission_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.ai.generativelanguage_v1beta3.services.permission_service.transports.PermissionServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.PermissionServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_permission', 'get_permission', 'list_permissions', 'update_permission', 'delete_permission', 'transfer_ownership')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_permission_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.ai.generativelanguage_v1beta3.services.permission_service.transports.PermissionServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.PermissionServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=(), quota_project_id='octopus')

def test_permission_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.ai.generativelanguage_v1beta3.services.permission_service.transports.PermissionServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.PermissionServiceTransport()
        adc.assert_called_once()

def test_permission_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        PermissionServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=(), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.PermissionServiceGrpcTransport, transports.PermissionServiceGrpcAsyncIOTransport])
def test_permission_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=(), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.PermissionServiceGrpcTransport, transports.PermissionServiceGrpcAsyncIOTransport, transports.PermissionServiceRestTransport])
def test_permission_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.PermissionServiceGrpcTransport, grpc_helpers), (transports.PermissionServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_permission_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('generativelanguage.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=(), scopes=['1', '2'], default_host='generativelanguage.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.PermissionServiceGrpcTransport, transports.PermissionServiceGrpcAsyncIOTransport])
def test_permission_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_permission_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.PermissionServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_permission_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='generativelanguage.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('generativelanguage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_permission_service_host_with_port(transport_name):
    if False:
        return 10
    client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='generativelanguage.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('generativelanguage.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_permission_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = PermissionServiceClient(credentials=creds1, transport=transport_name)
    client2 = PermissionServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_permission._session
    session2 = client2.transport.create_permission._session
    assert session1 != session2
    session1 = client1.transport.get_permission._session
    session2 = client2.transport.get_permission._session
    assert session1 != session2
    session1 = client1.transport.list_permissions._session
    session2 = client2.transport.list_permissions._session
    assert session1 != session2
    session1 = client1.transport.update_permission._session
    session2 = client2.transport.update_permission._session
    assert session1 != session2
    session1 = client1.transport.delete_permission._session
    session2 = client2.transport.delete_permission._session
    assert session1 != session2
    session1 = client1.transport.transfer_ownership._session
    session2 = client2.transport.transfer_ownership._session
    assert session1 != session2

def test_permission_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.PermissionServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_permission_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.PermissionServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.PermissionServiceGrpcTransport, transports.PermissionServiceGrpcAsyncIOTransport])
def test_permission_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('transport_class', [transports.PermissionServiceGrpcTransport, transports.PermissionServiceGrpcAsyncIOTransport])
def test_permission_service_transport_channel_mtls_with_adc(transport_class):
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

def test_permission_path():
    if False:
        return 10
    tuned_model = 'squid'
    permission = 'clam'
    expected = 'tunedModels/{tuned_model}/permissions/{permission}'.format(tuned_model=tuned_model, permission=permission)
    actual = PermissionServiceClient.permission_path(tuned_model, permission)
    assert expected == actual

def test_parse_permission_path():
    if False:
        i = 10
        return i + 15
    expected = {'tuned_model': 'whelk', 'permission': 'octopus'}
    path = PermissionServiceClient.permission_path(**expected)
    actual = PermissionServiceClient.parse_permission_path(path)
    assert expected == actual

def test_tuned_model_path():
    if False:
        i = 10
        return i + 15
    tuned_model = 'oyster'
    expected = 'tunedModels/{tuned_model}'.format(tuned_model=tuned_model)
    actual = PermissionServiceClient.tuned_model_path(tuned_model)
    assert expected == actual

def test_parse_tuned_model_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'tuned_model': 'nudibranch'}
    path = PermissionServiceClient.tuned_model_path(**expected)
    actual = PermissionServiceClient.parse_tuned_model_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = PermissionServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'mussel'}
    path = PermissionServiceClient.common_billing_account_path(**expected)
    actual = PermissionServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = PermissionServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = PermissionServiceClient.common_folder_path(**expected)
    actual = PermissionServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = PermissionServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = PermissionServiceClient.common_organization_path(**expected)
    actual = PermissionServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = PermissionServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam'}
    path = PermissionServiceClient.common_project_path(**expected)
    actual = PermissionServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = PermissionServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = PermissionServiceClient.common_location_path(**expected)
    actual = PermissionServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.PermissionServiceTransport, '_prep_wrapped_messages') as prep:
        client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.PermissionServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = PermissionServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = PermissionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = PermissionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(PermissionServiceClient, transports.PermissionServiceGrpcTransport), (PermissionServiceAsyncClient, transports.PermissionServiceGrpcAsyncIOTransport)])
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