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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.securesourcemanager_v1.services.secure_source_manager import SecureSourceManagerAsyncClient, SecureSourceManagerClient, pagers, transports
from google.cloud.securesourcemanager_v1.types import secure_source_manager

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert SecureSourceManagerClient._get_default_mtls_endpoint(None) is None
    assert SecureSourceManagerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SecureSourceManagerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SecureSourceManagerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SecureSourceManagerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SecureSourceManagerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SecureSourceManagerClient, 'grpc'), (SecureSourceManagerAsyncClient, 'grpc_asyncio'), (SecureSourceManagerClient, 'rest')])
def test_secure_source_manager_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('securesourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://securesourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SecureSourceManagerGrpcTransport, 'grpc'), (transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.SecureSourceManagerRestTransport, 'rest')])
def test_secure_source_manager_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(SecureSourceManagerClient, 'grpc'), (SecureSourceManagerAsyncClient, 'grpc_asyncio'), (SecureSourceManagerClient, 'rest')])
def test_secure_source_manager_client_from_service_account_file(client_class, transport_name):
    if False:
        i = 10
        return i + 15
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('securesourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://securesourcemanager.googleapis.com')

def test_secure_source_manager_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = SecureSourceManagerClient.get_transport_class()
    available_transports = [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerRestTransport]
    assert transport in available_transports
    transport = SecureSourceManagerClient.get_transport_class('grpc')
    assert transport == transports.SecureSourceManagerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport, 'grpc'), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (SecureSourceManagerClient, transports.SecureSourceManagerRestTransport, 'rest')])
@mock.patch.object(SecureSourceManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecureSourceManagerClient))
@mock.patch.object(SecureSourceManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecureSourceManagerAsyncClient))
def test_secure_source_manager_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(SecureSourceManagerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SecureSourceManagerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport, 'grpc', 'true'), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport, 'grpc', 'false'), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (SecureSourceManagerClient, transports.SecureSourceManagerRestTransport, 'rest', 'true'), (SecureSourceManagerClient, transports.SecureSourceManagerRestTransport, 'rest', 'false')])
@mock.patch.object(SecureSourceManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecureSourceManagerClient))
@mock.patch.object(SecureSourceManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecureSourceManagerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_secure_source_manager_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [SecureSourceManagerClient, SecureSourceManagerAsyncClient])
@mock.patch.object(SecureSourceManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecureSourceManagerClient))
@mock.patch.object(SecureSourceManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecureSourceManagerAsyncClient))
def test_secure_source_manager_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport, 'grpc'), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (SecureSourceManagerClient, transports.SecureSourceManagerRestTransport, 'rest')])
def test_secure_source_manager_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport, 'grpc', grpc_helpers), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (SecureSourceManagerClient, transports.SecureSourceManagerRestTransport, 'rest', None)])
def test_secure_source_manager_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_secure_source_manager_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.securesourcemanager_v1.services.secure_source_manager.transports.SecureSourceManagerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = SecureSourceManagerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport, 'grpc', grpc_helpers), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_secure_source_manager_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('securesourcemanager.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='securesourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [secure_source_manager.ListInstancesRequest, dict])
def test_list_instances(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = secure_source_manager.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_empty_call():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        client.list_instances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.ListInstancesRequest()

@pytest.mark.asyncio
async def test_list_instances_async(transport: str='grpc_asyncio', request_type=secure_source_manager.ListInstancesRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_instances_async_from_dict():
    await test_list_instances_async(request_type=dict)

def test_list_instances_field_headers():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = secure_source_manager.ListInstancesResponse()
        client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_instances_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.ListInstancesResponse())
        await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_instances_flattened():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = secure_source_manager.ListInstancesResponse()
        client.list_instances(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_instances_flattened_error():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_instances(secure_source_manager.ListInstancesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_instances_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = secure_source_manager.ListInstancesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.ListInstancesResponse())
        response = await client.list_instances(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_instances_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_instances(secure_source_manager.ListInstancesRequest(), parent='parent_value')

def test_list_instances_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance(), secure_source_manager.Instance()], next_page_token='abc'), secure_source_manager.ListInstancesResponse(instances=[], next_page_token='def'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance()], next_page_token='ghi'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_instances(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, secure_source_manager.Instance) for i in results))

def test_list_instances_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance(), secure_source_manager.Instance()], next_page_token='abc'), secure_source_manager.ListInstancesResponse(instances=[], next_page_token='def'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance()], next_page_token='ghi'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance()]), RuntimeError)
        pages = list(client.list_instances(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_instances_async_pager():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance(), secure_source_manager.Instance()], next_page_token='abc'), secure_source_manager.ListInstancesResponse(instances=[], next_page_token='def'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance()], next_page_token='ghi'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance()]), RuntimeError)
        async_pager = await client.list_instances(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, secure_source_manager.Instance) for i in responses))

@pytest.mark.asyncio
async def test_list_instances_async_pages():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance(), secure_source_manager.Instance()], next_page_token='abc'), secure_source_manager.ListInstancesResponse(instances=[], next_page_token='def'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance()], next_page_token='ghi'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_instances(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [secure_source_manager.GetInstanceRequest, dict])
def test_get_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = secure_source_manager.Instance(name='name_value', state=secure_source_manager.Instance.State.CREATING, state_note=secure_source_manager.Instance.StateNote.PAUSED_CMEK_UNAVAILABLE, kms_key='kms_key_value')
        response = client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.GetInstanceRequest()
    assert isinstance(response, secure_source_manager.Instance)
    assert response.name == 'name_value'
    assert response.state == secure_source_manager.Instance.State.CREATING
    assert response.state_note == secure_source_manager.Instance.StateNote.PAUSED_CMEK_UNAVAILABLE
    assert response.kms_key == 'kms_key_value'

def test_get_instance_empty_call():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        client.get_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.GetInstanceRequest()

@pytest.mark.asyncio
async def test_get_instance_async(transport: str='grpc_asyncio', request_type=secure_source_manager.GetInstanceRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.Instance(name='name_value', state=secure_source_manager.Instance.State.CREATING, state_note=secure_source_manager.Instance.StateNote.PAUSED_CMEK_UNAVAILABLE, kms_key='kms_key_value'))
        response = await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.GetInstanceRequest()
    assert isinstance(response, secure_source_manager.Instance)
    assert response.name == 'name_value'
    assert response.state == secure_source_manager.Instance.State.CREATING
    assert response.state_note == secure_source_manager.Instance.StateNote.PAUSED_CMEK_UNAVAILABLE
    assert response.kms_key == 'kms_key_value'

@pytest.mark.asyncio
async def test_get_instance_async_from_dict():
    await test_get_instance_async(request_type=dict)

def test_get_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = secure_source_manager.Instance()
        client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.Instance())
        await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_instance_flattened():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = secure_source_manager.Instance()
        client.get_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_instance_flattened_error():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance(secure_source_manager.GetInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = secure_source_manager.Instance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.Instance())
        response = await client.get_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance(secure_source_manager.GetInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [secure_source_manager.CreateInstanceRequest, dict])
def test_create_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.CreateInstanceRequest()
    assert isinstance(response, future.Future)

def test_create_instance_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        client.create_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.CreateInstanceRequest()

@pytest.mark.asyncio
async def test_create_instance_async(transport: str='grpc_asyncio', request_type=secure_source_manager.CreateInstanceRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.CreateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_instance_async_from_dict():
    await test_create_instance_async(request_type=dict)

def test_create_instance_field_headers():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.CreateInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_instance_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.CreateInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_instance_flattened():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(parent='parent_value', instance=secure_source_manager.Instance(name='name_value'), instance_id='instance_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = secure_source_manager.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val

def test_create_instance_flattened_error():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_instance(secure_source_manager.CreateInstanceRequest(), parent='parent_value', instance=secure_source_manager.Instance(name='name_value'), instance_id='instance_id_value')

@pytest.mark.asyncio
async def test_create_instance_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(parent='parent_value', instance=secure_source_manager.Instance(name='name_value'), instance_id='instance_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = secure_source_manager.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_instance_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_instance(secure_source_manager.CreateInstanceRequest(), parent='parent_value', instance=secure_source_manager.Instance(name='name_value'), instance_id='instance_id_value')

@pytest.mark.parametrize('request_type', [secure_source_manager.DeleteInstanceRequest, dict])
def test_delete_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

def test_delete_instance_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        client.delete_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.DeleteInstanceRequest()

@pytest.mark.asyncio
async def test_delete_instance_async(transport: str='grpc_asyncio', request_type=secure_source_manager.DeleteInstanceRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_instance_async_from_dict():
    await test_delete_instance_async(request_type=dict)

def test_delete_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.DeleteInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_instance_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.DeleteInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_instance(secure_source_manager.DeleteInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_instance_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_instance_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_instance(secure_source_manager.DeleteInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [secure_source_manager.ListRepositoriesRequest, dict])
def test_list_repositories(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = secure_source_manager.ListRepositoriesResponse(next_page_token='next_page_token_value')
        response = client.list_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.ListRepositoriesRequest()
    assert isinstance(response, pagers.ListRepositoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_repositories_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        client.list_repositories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.ListRepositoriesRequest()

@pytest.mark.asyncio
async def test_list_repositories_async(transport: str='grpc_asyncio', request_type=secure_source_manager.ListRepositoriesRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.ListRepositoriesResponse(next_page_token='next_page_token_value'))
        response = await client.list_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.ListRepositoriesRequest()
    assert isinstance(response, pagers.ListRepositoriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_repositories_async_from_dict():
    await test_list_repositories_async(request_type=dict)

def test_list_repositories_field_headers():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.ListRepositoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = secure_source_manager.ListRepositoriesResponse()
        client.list_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_repositories_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.ListRepositoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.ListRepositoriesResponse())
        await client.list_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_repositories_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = secure_source_manager.ListRepositoriesResponse()
        client.list_repositories(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_repositories_flattened_error():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_repositories(secure_source_manager.ListRepositoriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_repositories_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = secure_source_manager.ListRepositoriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.ListRepositoriesResponse())
        response = await client.list_repositories(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_repositories_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_repositories(secure_source_manager.ListRepositoriesRequest(), parent='parent_value')

def test_list_repositories_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.side_effect = (secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository(), secure_source_manager.Repository()], next_page_token='abc'), secure_source_manager.ListRepositoriesResponse(repositories=[], next_page_token='def'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository()], next_page_token='ghi'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_repositories(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, secure_source_manager.Repository) for i in results))

def test_list_repositories_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.side_effect = (secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository(), secure_source_manager.Repository()], next_page_token='abc'), secure_source_manager.ListRepositoriesResponse(repositories=[], next_page_token='def'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository()], next_page_token='ghi'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository()]), RuntimeError)
        pages = list(client.list_repositories(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_repositories_async_pager():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_repositories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository(), secure_source_manager.Repository()], next_page_token='abc'), secure_source_manager.ListRepositoriesResponse(repositories=[], next_page_token='def'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository()], next_page_token='ghi'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository()]), RuntimeError)
        async_pager = await client.list_repositories(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, secure_source_manager.Repository) for i in responses))

@pytest.mark.asyncio
async def test_list_repositories_async_pages():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_repositories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository(), secure_source_manager.Repository()], next_page_token='abc'), secure_source_manager.ListRepositoriesResponse(repositories=[], next_page_token='def'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository()], next_page_token='ghi'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_repositories(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [secure_source_manager.GetRepositoryRequest, dict])
def test_get_repository(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = secure_source_manager.Repository(name='name_value', description='description_value', instance='instance_value', uid='uid_value', etag='etag_value')
        response = client.get_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.GetRepositoryRequest()
    assert isinstance(response, secure_source_manager.Repository)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.instance == 'instance_value'
    assert response.uid == 'uid_value'
    assert response.etag == 'etag_value'

def test_get_repository_empty_call():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        client.get_repository()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.GetRepositoryRequest()

@pytest.mark.asyncio
async def test_get_repository_async(transport: str='grpc_asyncio', request_type=secure_source_manager.GetRepositoryRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.Repository(name='name_value', description='description_value', instance='instance_value', uid='uid_value', etag='etag_value'))
        response = await client.get_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.GetRepositoryRequest()
    assert isinstance(response, secure_source_manager.Repository)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.instance == 'instance_value'
    assert response.uid == 'uid_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_repository_async_from_dict():
    await test_get_repository_async(request_type=dict)

def test_get_repository_field_headers():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.GetRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = secure_source_manager.Repository()
        client.get_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_repository_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.GetRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.Repository())
        await client.get_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_repository_flattened():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = secure_source_manager.Repository()
        client.get_repository(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_repository_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_repository(secure_source_manager.GetRepositoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_repository_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = secure_source_manager.Repository()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(secure_source_manager.Repository())
        response = await client.get_repository(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_repository_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_repository(secure_source_manager.GetRepositoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [secure_source_manager.CreateRepositoryRequest, dict])
def test_create_repository(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.CreateRepositoryRequest()
    assert isinstance(response, future.Future)

def test_create_repository_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        client.create_repository()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.CreateRepositoryRequest()

@pytest.mark.asyncio
async def test_create_repository_async(transport: str='grpc_asyncio', request_type=secure_source_manager.CreateRepositoryRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.CreateRepositoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_repository_async_from_dict():
    await test_create_repository_async(request_type=dict)

def test_create_repository_field_headers():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.CreateRepositoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_repository_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.CreateRepositoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_repository_flattened():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_repository(parent='parent_value', repository=secure_source_manager.Repository(name='name_value'), repository_id='repository_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].repository
        mock_val = secure_source_manager.Repository(name='name_value')
        assert arg == mock_val
        arg = args[0].repository_id
        mock_val = 'repository_id_value'
        assert arg == mock_val

def test_create_repository_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_repository(secure_source_manager.CreateRepositoryRequest(), parent='parent_value', repository=secure_source_manager.Repository(name='name_value'), repository_id='repository_id_value')

@pytest.mark.asyncio
async def test_create_repository_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_repository(parent='parent_value', repository=secure_source_manager.Repository(name='name_value'), repository_id='repository_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].repository
        mock_val = secure_source_manager.Repository(name='name_value')
        assert arg == mock_val
        arg = args[0].repository_id
        mock_val = 'repository_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_repository_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_repository(secure_source_manager.CreateRepositoryRequest(), parent='parent_value', repository=secure_source_manager.Repository(name='name_value'), repository_id='repository_id_value')

@pytest.mark.parametrize('request_type', [secure_source_manager.DeleteRepositoryRequest, dict])
def test_delete_repository(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.DeleteRepositoryRequest()
    assert isinstance(response, future.Future)

def test_delete_repository_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        client.delete_repository()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.DeleteRepositoryRequest()

@pytest.mark.asyncio
async def test_delete_repository_async(transport: str='grpc_asyncio', request_type=secure_source_manager.DeleteRepositoryRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == secure_source_manager.DeleteRepositoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_repository_async_from_dict():
    await test_delete_repository_async(request_type=dict)

def test_delete_repository_field_headers():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.DeleteRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_repository_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = secure_source_manager.DeleteRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_repository_flattened():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_repository(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_repository_flattened_error():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_repository(secure_source_manager.DeleteRepositoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_repository_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_repository(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_repository_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_repository(secure_source_manager.DeleteRepositoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_repo(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy_repo(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_repo_empty_call():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        client.get_iam_policy_repo()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_repo_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy_repo(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_repo_async_from_dict():
    await test_get_iam_policy_repo_async(request_type=dict)

def test_get_iam_policy_repo_field_headers():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy_repo(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_repo_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy_repo(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_repo_from_dict_foreign():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy_repo(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_get_iam_policy_repo_flattened():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy_repo(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_get_iam_policy_repo_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_iam_policy_repo(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_get_iam_policy_repo_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy_repo(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_iam_policy_repo_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_iam_policy_repo(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_repo(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy_repo(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_repo_empty_call():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        client.set_iam_policy_repo()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_repo_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy_repo(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_repo_async_from_dict():
    await test_set_iam_policy_repo_async(request_type=dict)

def test_set_iam_policy_repo_field_headers():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy_repo(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_repo_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy_repo(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_repo_from_dict_foreign():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy_repo(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

def test_set_iam_policy_repo_flattened():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy_repo(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_set_iam_policy_repo_flattened_error():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_iam_policy_repo(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_set_iam_policy_repo_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy_repo), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy_repo(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_iam_policy_repo_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_iam_policy_repo(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_repo(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions_repo(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_repo_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        client.test_iam_permissions_repo()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_repo_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions_repo(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_repo_async_from_dict():
    await test_test_iam_permissions_repo_async(request_type=dict)

def test_test_iam_permissions_repo_field_headers():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions_repo(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_repo_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions_repo(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_repo_from_dict_foreign():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions_repo(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_test_iam_permissions_repo_flattened():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions_repo(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_test_iam_permissions_repo_flattened_error():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.test_iam_permissions_repo(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_test_iam_permissions_repo_flattened_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions_repo), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions_repo(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_test_iam_permissions_repo_flattened_error_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.test_iam_permissions_repo(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [secure_source_manager.ListInstancesRequest, dict])
def test_list_instances_rest(request_type):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_instances(request)
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_rest_required_fields(request_type=secure_source_manager.ListInstancesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = secure_source_manager.ListInstancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = secure_source_manager.ListInstancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_instances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_instances_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_instances_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_list_instances') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_list_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.ListInstancesRequest.pb(secure_source_manager.ListInstancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = secure_source_manager.ListInstancesResponse.to_json(secure_source_manager.ListInstancesResponse())
        request = secure_source_manager.ListInstancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = secure_source_manager.ListInstancesResponse()
        client.list_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_instances_rest_bad_request(transport: str='rest', request_type=secure_source_manager.ListInstancesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_instances(request)

def test_list_instances_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.ListInstancesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_list_instances_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_instances(secure_source_manager.ListInstancesRequest(), parent='parent_value')

def test_list_instances_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance(), secure_source_manager.Instance()], next_page_token='abc'), secure_source_manager.ListInstancesResponse(instances=[], next_page_token='def'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance()], next_page_token='ghi'), secure_source_manager.ListInstancesResponse(instances=[secure_source_manager.Instance(), secure_source_manager.Instance()]))
        response = response + response
        response = tuple((secure_source_manager.ListInstancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_instances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, secure_source_manager.Instance) for i in results))
        pages = list(client.list_instances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [secure_source_manager.GetInstanceRequest, dict])
def test_get_instance_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.Instance(name='name_value', state=secure_source_manager.Instance.State.CREATING, state_note=secure_source_manager.Instance.StateNote.PAUSED_CMEK_UNAVAILABLE, kms_key='kms_key_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance(request)
    assert isinstance(response, secure_source_manager.Instance)
    assert response.name == 'name_value'
    assert response.state == secure_source_manager.Instance.State.CREATING
    assert response.state_note == secure_source_manager.Instance.StateNote.PAUSED_CMEK_UNAVAILABLE
    assert response.kms_key == 'kms_key_value'

def test_get_instance_rest_required_fields(request_type=secure_source_manager.GetInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = secure_source_manager.Instance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = secure_source_manager.Instance.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_get_instance') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_get_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.GetInstanceRequest.pb(secure_source_manager.GetInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = secure_source_manager.Instance.to_json(secure_source_manager.Instance())
        request = secure_source_manager.GetInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = secure_source_manager.Instance()
        client.get_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_rest_bad_request(transport: str='rest', request_type=secure_source_manager.GetInstanceRequest):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_instance(request)

def test_get_instance_rest_flattened():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.Instance()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_get_instance_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance(secure_source_manager.GetInstanceRequest(), name='name_value')

def test_get_instance_rest_error():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [secure_source_manager.CreateInstanceRequest, dict])
def test_create_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['instance'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'state': 1, 'state_note': 1, 'kms_key': 'kms_key_value', 'host_config': {'html': 'html_value', 'api': 'api_value', 'git_http': 'git_http_value', 'git_ssh': 'git_ssh_value'}}
    test_field = secure_source_manager.CreateInstanceRequest.meta.fields['instance']

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
    for (field, value) in request_init['instance'].items():
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
                for i in range(0, len(request_init['instance'][field])):
                    del request_init['instance'][field][i][subfield]
            else:
                del request_init['instance'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_instance(request)
    assert response.operation.name == 'operations/spam'

def test_create_instance_rest_required_fields(request_type=secure_source_manager.CreateInstanceRequest):
    if False:
        return 10
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['instance_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'instanceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == request_init['instance_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['instanceId'] = 'instance_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('instance_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == 'instance_id_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_instance(request)
            expected_params = [('instanceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('instanceId', 'requestId')) & set(('parent', 'instanceId', 'instance'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_create_instance') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_create_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.CreateInstanceRequest.pb(secure_source_manager.CreateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = secure_source_manager.CreateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_instance_rest_bad_request(transport: str='rest', request_type=secure_source_manager.CreateInstanceRequest):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_instance(request)

def test_create_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', instance=secure_source_manager.Instance(name='name_value'), instance_id='instance_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_create_instance_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_instance(secure_source_manager.CreateInstanceRequest(), parent='parent_value', instance=secure_source_manager.Instance(name='name_value'), instance_id='instance_id_value')

def test_create_instance_rest_error():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [secure_source_manager.DeleteInstanceRequest, dict])
def test_delete_instance_rest(request_type):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_instance(request)
    assert response.operation.name == 'operations/spam'

def test_delete_instance_rest_required_fields(request_type=secure_source_manager.DeleteInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_delete_instance') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_delete_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.DeleteInstanceRequest.pb(secure_source_manager.DeleteInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = secure_source_manager.DeleteInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_instance_rest_bad_request(transport: str='rest', request_type=secure_source_manager.DeleteInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_instance(request)

def test_delete_instance_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_delete_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_instance(secure_source_manager.DeleteInstanceRequest(), name='name_value')

def test_delete_instance_rest_error():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [secure_source_manager.ListRepositoriesRequest, dict])
def test_list_repositories_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.ListRepositoriesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.ListRepositoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_repositories(request)
    assert isinstance(response, pagers.ListRepositoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_repositories_rest_required_fields(request_type=secure_source_manager.ListRepositoriesRequest):
    if False:
        return 10
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_repositories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_repositories._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = secure_source_manager.ListRepositoriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = secure_source_manager.ListRepositoriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_repositories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_repositories_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_repositories._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_repositories_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_list_repositories') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_list_repositories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.ListRepositoriesRequest.pb(secure_source_manager.ListRepositoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = secure_source_manager.ListRepositoriesResponse.to_json(secure_source_manager.ListRepositoriesResponse())
        request = secure_source_manager.ListRepositoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = secure_source_manager.ListRepositoriesResponse()
        client.list_repositories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_repositories_rest_bad_request(transport: str='rest', request_type=secure_source_manager.ListRepositoriesRequest):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_repositories(request)

def test_list_repositories_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.ListRepositoriesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.ListRepositoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_repositories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/repositories' % client.transport._host, args[1])

def test_list_repositories_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_repositories(secure_source_manager.ListRepositoriesRequest(), parent='parent_value')

def test_list_repositories_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository(), secure_source_manager.Repository()], next_page_token='abc'), secure_source_manager.ListRepositoriesResponse(repositories=[], next_page_token='def'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository()], next_page_token='ghi'), secure_source_manager.ListRepositoriesResponse(repositories=[secure_source_manager.Repository(), secure_source_manager.Repository()]))
        response = response + response
        response = tuple((secure_source_manager.ListRepositoriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_repositories(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, secure_source_manager.Repository) for i in results))
        pages = list(client.list_repositories(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [secure_source_manager.GetRepositoryRequest, dict])
def test_get_repository_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.Repository(name='name_value', description='description_value', instance='instance_value', uid='uid_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.Repository.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_repository(request)
    assert isinstance(response, secure_source_manager.Repository)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.instance == 'instance_value'
    assert response.uid == 'uid_value'
    assert response.etag == 'etag_value'

def test_get_repository_rest_required_fields(request_type=secure_source_manager.GetRepositoryRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = secure_source_manager.Repository()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = secure_source_manager.Repository.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_repository(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_repository_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_repository._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_repository_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_get_repository') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_get_repository') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.GetRepositoryRequest.pb(secure_source_manager.GetRepositoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = secure_source_manager.Repository.to_json(secure_source_manager.Repository())
        request = secure_source_manager.GetRepositoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = secure_source_manager.Repository()
        client.get_repository(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_repository_rest_bad_request(transport: str='rest', request_type=secure_source_manager.GetRepositoryRequest):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_repository(request)

def test_get_repository_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = secure_source_manager.Repository()
        sample_request = {'name': 'projects/sample1/locations/sample2/repositories/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = secure_source_manager.Repository.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_repository(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/repositories/*}' % client.transport._host, args[1])

def test_get_repository_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_repository(secure_source_manager.GetRepositoryRequest(), name='name_value')

def test_get_repository_rest_error():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [secure_source_manager.CreateRepositoryRequest, dict])
def test_create_repository_rest(request_type):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['repository'] = {'name': 'name_value', 'description': 'description_value', 'instance': 'instance_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'uris': {'html': 'html_value', 'git_https': 'git_https_value', 'api': 'api_value'}, 'initial_config': {'default_branch': 'default_branch_value', 'gitignores': ['gitignores_value1', 'gitignores_value2'], 'license_': 'license__value', 'readme': 'readme_value'}}
    test_field = secure_source_manager.CreateRepositoryRequest.meta.fields['repository']

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
    for (field, value) in request_init['repository'].items():
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
                for i in range(0, len(request_init['repository'][field])):
                    del request_init['repository'][field][i][subfield]
            else:
                del request_init['repository'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_repository(request)
    assert response.operation.name == 'operations/spam'

def test_create_repository_rest_required_fields(request_type=secure_source_manager.CreateRepositoryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['repository_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'repositoryId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'repositoryId' in jsonified_request
    assert jsonified_request['repositoryId'] == request_init['repository_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['repositoryId'] = 'repository_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_repository._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('repository_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'repositoryId' in jsonified_request
    assert jsonified_request['repositoryId'] == 'repository_id_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_repository(request)
            expected_params = [('repositoryId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_repository_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_repository._get_unset_required_fields({})
    assert set(unset_fields) == set(('repositoryId',)) & set(('parent', 'repository', 'repositoryId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_repository_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_create_repository') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_create_repository') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.CreateRepositoryRequest.pb(secure_source_manager.CreateRepositoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = secure_source_manager.CreateRepositoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_repository(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_repository_rest_bad_request(transport: str='rest', request_type=secure_source_manager.CreateRepositoryRequest):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_repository(request)

def test_create_repository_rest_flattened():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', repository=secure_source_manager.Repository(name='name_value'), repository_id='repository_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_repository(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/repositories' % client.transport._host, args[1])

def test_create_repository_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_repository(secure_source_manager.CreateRepositoryRequest(), parent='parent_value', repository=secure_source_manager.Repository(name='name_value'), repository_id='repository_id_value')

def test_create_repository_rest_error():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [secure_source_manager.DeleteRepositoryRequest, dict])
def test_delete_repository_rest(request_type):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_repository(request)
    assert response.operation.name == 'operations/spam'

def test_delete_repository_rest_required_fields(request_type=secure_source_manager.DeleteRepositoryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_repository._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_repository(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_repository_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_repository._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_repository_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_delete_repository') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_delete_repository') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = secure_source_manager.DeleteRepositoryRequest.pb(secure_source_manager.DeleteRepositoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = secure_source_manager.DeleteRepositoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_repository(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_repository_rest_bad_request(transport: str='rest', request_type=secure_source_manager.DeleteRepositoryRequest):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_repository(request)

def test_delete_repository_rest_flattened():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/repositories/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_repository(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/repositories/*}' % client.transport._host, args[1])

def test_delete_repository_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_repository(secure_source_manager.DeleteRepositoryRequest(), name='name_value')

def test_delete_repository_rest_error():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_repo_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy_repo(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_repo_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy_repo._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy_repo._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options',))
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy_repo(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_repo_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy_repo._get_unset_required_fields({})
    assert set(unset_fields) == set(('options',)) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_repo_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_get_iam_policy_repo') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_get_iam_policy_repo') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy_repo(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_repo_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy_repo(request)

def test_get_iam_policy_repo_rest_flattened():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy_repo(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=projects/*/locations/*/repositories/*}:getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_repo_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy_repo(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

def test_get_iam_policy_repo_rest_error():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_repo_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy_repo(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_repo_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        return 10
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy_repo._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy_repo._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy_repo(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_repo_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy_repo._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_repo_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_set_iam_policy_repo') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_set_iam_policy_repo') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy_repo(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_repo_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy_repo(request)

def test_set_iam_policy_repo_rest_flattened():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy_repo(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=projects/*/locations/*/repositories/*}:setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_repo_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy_repo(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

def test_set_iam_policy_repo_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_repo_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions_repo(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_repo_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecureSourceManagerRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions_repo._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions_repo._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions_repo(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_repo_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions_repo._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_repo_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecureSourceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecureSourceManagerRestInterceptor())
    client = SecureSourceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'post_test_iam_permissions_repo') as post, mock.patch.object(transports.SecureSourceManagerRestInterceptor, 'pre_test_iam_permissions_repo') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions_repo(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_repo_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions_repo(request)

def test_test_iam_permissions_repo_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        sample_request = {'resource': 'projects/sample1/locations/sample2/repositories/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions_repo(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=projects/*/locations/*/repositories/*}:testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_repo_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions_repo(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value')

def test_test_iam_permissions_repo_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.SecureSourceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SecureSourceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecureSourceManagerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SecureSourceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SecureSourceManagerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SecureSourceManagerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SecureSourceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecureSourceManagerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.SecureSourceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SecureSourceManagerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecureSourceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.SecureSourceManagerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerGrpcAsyncIOTransport, transports.SecureSourceManagerRestTransport])
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
        for i in range(10):
            print('nop')
    transport = SecureSourceManagerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.SecureSourceManagerGrpcTransport)

def test_secure_source_manager_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SecureSourceManagerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_secure_source_manager_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.securesourcemanager_v1.services.secure_source_manager.transports.SecureSourceManagerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SecureSourceManagerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_instances', 'get_instance', 'create_instance', 'delete_instance', 'list_repositories', 'get_repository', 'create_repository', 'delete_repository', 'get_iam_policy_repo', 'set_iam_policy_repo', 'test_iam_permissions_repo', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_secure_source_manager_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.securesourcemanager_v1.services.secure_source_manager.transports.SecureSourceManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SecureSourceManagerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_secure_source_manager_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.securesourcemanager_v1.services.secure_source_manager.transports.SecureSourceManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SecureSourceManagerTransport()
        adc.assert_called_once()

def test_secure_source_manager_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SecureSourceManagerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerGrpcAsyncIOTransport])
def test_secure_source_manager_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerGrpcAsyncIOTransport, transports.SecureSourceManagerRestTransport])
def test_secure_source_manager_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.SecureSourceManagerGrpcTransport, grpc_helpers), (transports.SecureSourceManagerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_secure_source_manager_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('securesourcemanager.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='securesourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerGrpcAsyncIOTransport])
def test_secure_source_manager_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_secure_source_manager_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.SecureSourceManagerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_secure_source_manager_rest_lro_client():
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_secure_source_manager_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='securesourcemanager.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('securesourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://securesourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_secure_source_manager_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='securesourcemanager.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('securesourcemanager.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://securesourcemanager.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_secure_source_manager_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = SecureSourceManagerClient(credentials=creds1, transport=transport_name)
    client2 = SecureSourceManagerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_instances._session
    session2 = client2.transport.list_instances._session
    assert session1 != session2
    session1 = client1.transport.get_instance._session
    session2 = client2.transport.get_instance._session
    assert session1 != session2
    session1 = client1.transport.create_instance._session
    session2 = client2.transport.create_instance._session
    assert session1 != session2
    session1 = client1.transport.delete_instance._session
    session2 = client2.transport.delete_instance._session
    assert session1 != session2
    session1 = client1.transport.list_repositories._session
    session2 = client2.transport.list_repositories._session
    assert session1 != session2
    session1 = client1.transport.get_repository._session
    session2 = client2.transport.get_repository._session
    assert session1 != session2
    session1 = client1.transport.create_repository._session
    session2 = client2.transport.create_repository._session
    assert session1 != session2
    session1 = client1.transport.delete_repository._session
    session2 = client2.transport.delete_repository._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy_repo._session
    session2 = client2.transport.get_iam_policy_repo._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy_repo._session
    session2 = client2.transport.set_iam_policy_repo._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions_repo._session
    session2 = client2.transport.test_iam_permissions_repo._session
    assert session1 != session2

def test_secure_source_manager_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SecureSourceManagerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_secure_source_manager_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SecureSourceManagerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerGrpcAsyncIOTransport])
def test_secure_source_manager_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.SecureSourceManagerGrpcTransport, transports.SecureSourceManagerGrpcAsyncIOTransport])
def test_secure_source_manager_transport_channel_mtls_with_adc(transport_class):
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

def test_secure_source_manager_grpc_lro_client():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_secure_source_manager_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_crypto_key_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    key_ring = 'whelk'
    crypto_key = 'octopus'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key)
    actual = SecureSourceManagerClient.crypto_key_path(project, location, key_ring, crypto_key)
    assert expected == actual

def test_parse_crypto_key_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'key_ring': 'cuttlefish', 'crypto_key': 'mussel'}
    path = SecureSourceManagerClient.crypto_key_path(**expected)
    actual = SecureSourceManagerClient.parse_crypto_key_path(path)
    assert expected == actual

def test_instance_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    instance = 'scallop'
    expected = 'projects/{project}/locations/{location}/instances/{instance}'.format(project=project, location=location, instance=instance)
    actual = SecureSourceManagerClient.instance_path(project, location, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        return 10
    expected = {'project': 'abalone', 'location': 'squid', 'instance': 'clam'}
    path = SecureSourceManagerClient.instance_path(**expected)
    actual = SecureSourceManagerClient.parse_instance_path(path)
    assert expected == actual

def test_repository_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    repository = 'oyster'
    expected = 'projects/{project}/locations/{location}/repositories/{repository}'.format(project=project, location=location, repository=repository)
    actual = SecureSourceManagerClient.repository_path(project, location, repository)
    assert expected == actual

def test_parse_repository_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'repository': 'mussel'}
    path = SecureSourceManagerClient.repository_path(**expected)
    actual = SecureSourceManagerClient.parse_repository_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SecureSourceManagerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'nautilus'}
    path = SecureSourceManagerClient.common_billing_account_path(**expected)
    actual = SecureSourceManagerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SecureSourceManagerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'abalone'}
    path = SecureSourceManagerClient.common_folder_path(**expected)
    actual = SecureSourceManagerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SecureSourceManagerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'clam'}
    path = SecureSourceManagerClient.common_organization_path(**expected)
    actual = SecureSourceManagerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = SecureSourceManagerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus'}
    path = SecureSourceManagerClient.common_project_path(**expected)
    actual = SecureSourceManagerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SecureSourceManagerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = SecureSourceManagerClient.common_location_path(**expected)
    actual = SecureSourceManagerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SecureSourceManagerTransport, '_prep_wrapped_messages') as prep:
        client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SecureSourceManagerTransport, '_prep_wrapped_messages') as prep:
        transport_class = SecureSourceManagerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/instances/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/instances/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/instances/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio'):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_set_iam_policy_from_dict():
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio'):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_field_headers():
    if False:
        return 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_get_iam_policy_from_dict():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio'):
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_test_iam_permissions_from_dict():
    if False:
        while True:
            i = 10
    client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = SecureSourceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = SecureSourceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SecureSourceManagerClient, transports.SecureSourceManagerGrpcTransport), (SecureSourceManagerAsyncClient, transports.SecureSourceManagerGrpcAsyncIOTransport)])
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