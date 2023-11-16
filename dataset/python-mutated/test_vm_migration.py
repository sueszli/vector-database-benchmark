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
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.vmmigration_v1.services.vm_migration import VmMigrationAsyncClient, VmMigrationClient, pagers, transports
from google.cloud.vmmigration_v1.types import vmmigration

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert VmMigrationClient._get_default_mtls_endpoint(None) is None
    assert VmMigrationClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert VmMigrationClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert VmMigrationClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert VmMigrationClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert VmMigrationClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(VmMigrationClient, 'grpc'), (VmMigrationAsyncClient, 'grpc_asyncio'), (VmMigrationClient, 'rest')])
def test_vm_migration_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('vmmigration.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmmigration.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.VmMigrationGrpcTransport, 'grpc'), (transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.VmMigrationRestTransport, 'rest')])
def test_vm_migration_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(VmMigrationClient, 'grpc'), (VmMigrationAsyncClient, 'grpc_asyncio'), (VmMigrationClient, 'rest')])
def test_vm_migration_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('vmmigration.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmmigration.googleapis.com')

def test_vm_migration_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = VmMigrationClient.get_transport_class()
    available_transports = [transports.VmMigrationGrpcTransport, transports.VmMigrationRestTransport]
    assert transport in available_transports
    transport = VmMigrationClient.get_transport_class('grpc')
    assert transport == transports.VmMigrationGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VmMigrationClient, transports.VmMigrationGrpcTransport, 'grpc'), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio'), (VmMigrationClient, transports.VmMigrationRestTransport, 'rest')])
@mock.patch.object(VmMigrationClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmMigrationClient))
@mock.patch.object(VmMigrationAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmMigrationAsyncClient))
def test_vm_migration_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(VmMigrationClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(VmMigrationClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(VmMigrationClient, transports.VmMigrationGrpcTransport, 'grpc', 'true'), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (VmMigrationClient, transports.VmMigrationGrpcTransport, 'grpc', 'false'), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (VmMigrationClient, transports.VmMigrationRestTransport, 'rest', 'true'), (VmMigrationClient, transports.VmMigrationRestTransport, 'rest', 'false')])
@mock.patch.object(VmMigrationClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmMigrationClient))
@mock.patch.object(VmMigrationAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmMigrationAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_vm_migration_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [VmMigrationClient, VmMigrationAsyncClient])
@mock.patch.object(VmMigrationClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmMigrationClient))
@mock.patch.object(VmMigrationAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmMigrationAsyncClient))
def test_vm_migration_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VmMigrationClient, transports.VmMigrationGrpcTransport, 'grpc'), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio'), (VmMigrationClient, transports.VmMigrationRestTransport, 'rest')])
def test_vm_migration_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VmMigrationClient, transports.VmMigrationGrpcTransport, 'grpc', grpc_helpers), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (VmMigrationClient, transports.VmMigrationRestTransport, 'rest', None)])
def test_vm_migration_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_vm_migration_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.vmmigration_v1.services.vm_migration.transports.VmMigrationGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = VmMigrationClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VmMigrationClient, transports.VmMigrationGrpcTransport, 'grpc', grpc_helpers), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_vm_migration_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
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
        create_channel.assert_called_with('vmmigration.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='vmmigration.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [vmmigration.ListSourcesRequest, dict])
def test_list_sources(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.return_value = vmmigration.ListSourcesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_sources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListSourcesRequest()
    assert isinstance(response, pagers.ListSourcesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_sources_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        client.list_sources()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListSourcesRequest()

@pytest.mark.asyncio
async def test_list_sources_async(transport: str='grpc_asyncio', request_type=vmmigration.ListSourcesRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListSourcesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_sources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListSourcesRequest()
    assert isinstance(response, pagers.ListSourcesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_sources_async_from_dict():
    await test_list_sources_async(request_type=dict)

def test_list_sources_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListSourcesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.return_value = vmmigration.ListSourcesResponse()
        client.list_sources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_sources_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListSourcesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListSourcesResponse())
        await client.list_sources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_sources_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.return_value = vmmigration.ListSourcesResponse()
        client.list_sources(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_sources_flattened_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_sources(vmmigration.ListSourcesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_sources_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.return_value = vmmigration.ListSourcesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListSourcesResponse())
        response = await client.list_sources(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_sources_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_sources(vmmigration.ListSourcesRequest(), parent='parent_value')

def test_list_sources_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.side_effect = (vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source(), vmmigration.Source()], next_page_token='abc'), vmmigration.ListSourcesResponse(sources=[], next_page_token='def'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source()], next_page_token='ghi'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_sources(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.Source) for i in results))

def test_list_sources_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_sources), '__call__') as call:
        call.side_effect = (vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source(), vmmigration.Source()], next_page_token='abc'), vmmigration.ListSourcesResponse(sources=[], next_page_token='def'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source()], next_page_token='ghi'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source()]), RuntimeError)
        pages = list(client.list_sources(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_sources_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_sources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source(), vmmigration.Source()], next_page_token='abc'), vmmigration.ListSourcesResponse(sources=[], next_page_token='def'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source()], next_page_token='ghi'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source()]), RuntimeError)
        async_pager = await client.list_sources(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.Source) for i in responses))

@pytest.mark.asyncio
async def test_list_sources_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_sources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source(), vmmigration.Source()], next_page_token='abc'), vmmigration.ListSourcesResponse(sources=[], next_page_token='def'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source()], next_page_token='ghi'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_sources(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetSourceRequest, dict])
def test_get_source(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        call.return_value = vmmigration.Source(name='name_value', description='description_value')
        response = client.get_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetSourceRequest()
    assert isinstance(response, vmmigration.Source)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

def test_get_source_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        client.get_source()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetSourceRequest()

@pytest.mark.asyncio
async def test_get_source_async(transport: str='grpc_asyncio', request_type=vmmigration.GetSourceRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.Source(name='name_value', description='description_value'))
        response = await client.get_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetSourceRequest()
    assert isinstance(response, vmmigration.Source)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_get_source_async_from_dict():
    await test_get_source_async(request_type=dict)

def test_get_source_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetSourceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        call.return_value = vmmigration.Source()
        client.get_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_source_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetSourceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.Source())
        await client.get_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_source_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        call.return_value = vmmigration.Source()
        client.get_source(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_source_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_source(vmmigration.GetSourceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_source_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_source), '__call__') as call:
        call.return_value = vmmigration.Source()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.Source())
        response = await client.get_source(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_source_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_source(vmmigration.GetSourceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateSourceRequest, dict])
def test_create_source(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateSourceRequest()
    assert isinstance(response, future.Future)

def test_create_source_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        client.create_source()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateSourceRequest()

@pytest.mark.asyncio
async def test_create_source_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateSourceRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateSourceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_source_async_from_dict():
    await test_create_source_async(request_type=dict)

def test_create_source_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateSourceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_source_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateSourceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_source_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_source(parent='parent_value', source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), source_id='source_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].source
        mock_val = vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value'))
        assert arg == mock_val
        arg = args[0].source_id
        mock_val = 'source_id_value'
        assert arg == mock_val

def test_create_source_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_source(vmmigration.CreateSourceRequest(), parent='parent_value', source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), source_id='source_id_value')

@pytest.mark.asyncio
async def test_create_source_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_source(parent='parent_value', source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), source_id='source_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].source
        mock_val = vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value'))
        assert arg == mock_val
        arg = args[0].source_id
        mock_val = 'source_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_source_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_source(vmmigration.CreateSourceRequest(), parent='parent_value', source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), source_id='source_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateSourceRequest, dict])
def test_update_source(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateSourceRequest()
    assert isinstance(response, future.Future)

def test_update_source_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        client.update_source()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateSourceRequest()

@pytest.mark.asyncio
async def test_update_source_async(transport: str='grpc_asyncio', request_type=vmmigration.UpdateSourceRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateSourceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_source_async_from_dict():
    await test_update_source_async(request_type=dict)

def test_update_source_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateSourceRequest()
    request.source.name = 'name_value'
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'source.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_source_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateSourceRequest()
    request.source.name = 'name_value'
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'source.name=name_value') in kw['metadata']

def test_update_source_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_source(source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].source
        mock_val = vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_source_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_source(vmmigration.UpdateSourceRequest(), source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_source_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_source(source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].source
        mock_val = vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_source_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_source(vmmigration.UpdateSourceRequest(), source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmmigration.DeleteSourceRequest, dict])
def test_delete_source(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteSourceRequest()
    assert isinstance(response, future.Future)

def test_delete_source_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        client.delete_source()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteSourceRequest()

@pytest.mark.asyncio
async def test_delete_source_async(transport: str='grpc_asyncio', request_type=vmmigration.DeleteSourceRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteSourceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_source_async_from_dict():
    await test_delete_source_async(request_type=dict)

def test_delete_source_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteSourceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_source_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteSourceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_source_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_source(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_source_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_source(vmmigration.DeleteSourceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_source_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_source(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_source_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_source(vmmigration.DeleteSourceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.FetchInventoryRequest, dict])
def test_fetch_inventory(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        call.return_value = vmmigration.FetchInventoryResponse()
        response = client.fetch_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.FetchInventoryRequest()
    assert isinstance(response, vmmigration.FetchInventoryResponse)

def test_fetch_inventory_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        client.fetch_inventory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.FetchInventoryRequest()

@pytest.mark.asyncio
async def test_fetch_inventory_async(transport: str='grpc_asyncio', request_type=vmmigration.FetchInventoryRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.FetchInventoryResponse())
        response = await client.fetch_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.FetchInventoryRequest()
    assert isinstance(response, vmmigration.FetchInventoryResponse)

@pytest.mark.asyncio
async def test_fetch_inventory_async_from_dict():
    await test_fetch_inventory_async(request_type=dict)

def test_fetch_inventory_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.FetchInventoryRequest()
    request.source = 'source_value'
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        call.return_value = vmmigration.FetchInventoryResponse()
        client.fetch_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'source=source_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_inventory_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.FetchInventoryRequest()
    request.source = 'source_value'
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.FetchInventoryResponse())
        await client.fetch_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'source=source_value') in kw['metadata']

def test_fetch_inventory_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        call.return_value = vmmigration.FetchInventoryResponse()
        client.fetch_inventory(source='source_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].source
        mock_val = 'source_value'
        assert arg == mock_val

def test_fetch_inventory_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.fetch_inventory(vmmigration.FetchInventoryRequest(), source='source_value')

@pytest.mark.asyncio
async def test_fetch_inventory_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_inventory), '__call__') as call:
        call.return_value = vmmigration.FetchInventoryResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.FetchInventoryResponse())
        response = await client.fetch_inventory(source='source_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].source
        mock_val = 'source_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_fetch_inventory_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.fetch_inventory(vmmigration.FetchInventoryRequest(), source='source_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListUtilizationReportsRequest, dict])
def test_list_utilization_reports(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.return_value = vmmigration.ListUtilizationReportsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_utilization_reports(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListUtilizationReportsRequest()
    assert isinstance(response, pagers.ListUtilizationReportsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_utilization_reports_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        client.list_utilization_reports()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListUtilizationReportsRequest()

@pytest.mark.asyncio
async def test_list_utilization_reports_async(transport: str='grpc_asyncio', request_type=vmmigration.ListUtilizationReportsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListUtilizationReportsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_utilization_reports(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListUtilizationReportsRequest()
    assert isinstance(response, pagers.ListUtilizationReportsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_utilization_reports_async_from_dict():
    await test_list_utilization_reports_async(request_type=dict)

def test_list_utilization_reports_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListUtilizationReportsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.return_value = vmmigration.ListUtilizationReportsResponse()
        client.list_utilization_reports(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_utilization_reports_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListUtilizationReportsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListUtilizationReportsResponse())
        await client.list_utilization_reports(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_utilization_reports_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.return_value = vmmigration.ListUtilizationReportsResponse()
        client.list_utilization_reports(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_utilization_reports_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_utilization_reports(vmmigration.ListUtilizationReportsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_utilization_reports_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.return_value = vmmigration.ListUtilizationReportsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListUtilizationReportsResponse())
        response = await client.list_utilization_reports(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_utilization_reports_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_utilization_reports(vmmigration.ListUtilizationReportsRequest(), parent='parent_value')

def test_list_utilization_reports_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.side_effect = (vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport(), vmmigration.UtilizationReport()], next_page_token='abc'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[], next_page_token='def'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport()], next_page_token='ghi'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_utilization_reports(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.UtilizationReport) for i in results))

def test_list_utilization_reports_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__') as call:
        call.side_effect = (vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport(), vmmigration.UtilizationReport()], next_page_token='abc'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[], next_page_token='def'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport()], next_page_token='ghi'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport()]), RuntimeError)
        pages = list(client.list_utilization_reports(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_utilization_reports_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport(), vmmigration.UtilizationReport()], next_page_token='abc'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[], next_page_token='def'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport()], next_page_token='ghi'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport()]), RuntimeError)
        async_pager = await client.list_utilization_reports(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.UtilizationReport) for i in responses))

@pytest.mark.asyncio
async def test_list_utilization_reports_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_utilization_reports), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport(), vmmigration.UtilizationReport()], next_page_token='abc'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[], next_page_token='def'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport()], next_page_token='ghi'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_utilization_reports(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetUtilizationReportRequest, dict])
def test_get_utilization_report(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        call.return_value = vmmigration.UtilizationReport(name='name_value', display_name='display_name_value', state=vmmigration.UtilizationReport.State.CREATING, time_frame=vmmigration.UtilizationReport.TimeFrame.WEEK, vm_count=875)
        response = client.get_utilization_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetUtilizationReportRequest()
    assert isinstance(response, vmmigration.UtilizationReport)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == vmmigration.UtilizationReport.State.CREATING
    assert response.time_frame == vmmigration.UtilizationReport.TimeFrame.WEEK
    assert response.vm_count == 875

def test_get_utilization_report_empty_call():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        client.get_utilization_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetUtilizationReportRequest()

@pytest.mark.asyncio
async def test_get_utilization_report_async(transport: str='grpc_asyncio', request_type=vmmigration.GetUtilizationReportRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.UtilizationReport(name='name_value', display_name='display_name_value', state=vmmigration.UtilizationReport.State.CREATING, time_frame=vmmigration.UtilizationReport.TimeFrame.WEEK, vm_count=875))
        response = await client.get_utilization_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetUtilizationReportRequest()
    assert isinstance(response, vmmigration.UtilizationReport)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == vmmigration.UtilizationReport.State.CREATING
    assert response.time_frame == vmmigration.UtilizationReport.TimeFrame.WEEK
    assert response.vm_count == 875

@pytest.mark.asyncio
async def test_get_utilization_report_async_from_dict():
    await test_get_utilization_report_async(request_type=dict)

def test_get_utilization_report_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetUtilizationReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        call.return_value = vmmigration.UtilizationReport()
        client.get_utilization_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_utilization_report_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetUtilizationReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.UtilizationReport())
        await client.get_utilization_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_utilization_report_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        call.return_value = vmmigration.UtilizationReport()
        client.get_utilization_report(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_utilization_report_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_utilization_report(vmmigration.GetUtilizationReportRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_utilization_report_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_utilization_report), '__call__') as call:
        call.return_value = vmmigration.UtilizationReport()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.UtilizationReport())
        response = await client.get_utilization_report(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_utilization_report_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_utilization_report(vmmigration.GetUtilizationReportRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateUtilizationReportRequest, dict])
def test_create_utilization_report(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_utilization_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateUtilizationReportRequest()
    assert isinstance(response, future.Future)

def test_create_utilization_report_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        client.create_utilization_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateUtilizationReportRequest()

@pytest.mark.asyncio
async def test_create_utilization_report_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateUtilizationReportRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_utilization_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateUtilizationReportRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_utilization_report_async_from_dict():
    await test_create_utilization_report_async(request_type=dict)

def test_create_utilization_report_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateUtilizationReportRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_utilization_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_utilization_report_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateUtilizationReportRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_utilization_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_utilization_report_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_utilization_report(parent='parent_value', utilization_report=vmmigration.UtilizationReport(name='name_value'), utilization_report_id='utilization_report_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].utilization_report
        mock_val = vmmigration.UtilizationReport(name='name_value')
        assert arg == mock_val
        arg = args[0].utilization_report_id
        mock_val = 'utilization_report_id_value'
        assert arg == mock_val

def test_create_utilization_report_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_utilization_report(vmmigration.CreateUtilizationReportRequest(), parent='parent_value', utilization_report=vmmigration.UtilizationReport(name='name_value'), utilization_report_id='utilization_report_id_value')

@pytest.mark.asyncio
async def test_create_utilization_report_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_utilization_report(parent='parent_value', utilization_report=vmmigration.UtilizationReport(name='name_value'), utilization_report_id='utilization_report_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].utilization_report
        mock_val = vmmigration.UtilizationReport(name='name_value')
        assert arg == mock_val
        arg = args[0].utilization_report_id
        mock_val = 'utilization_report_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_utilization_report_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_utilization_report(vmmigration.CreateUtilizationReportRequest(), parent='parent_value', utilization_report=vmmigration.UtilizationReport(name='name_value'), utilization_report_id='utilization_report_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteUtilizationReportRequest, dict])
def test_delete_utilization_report(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_utilization_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteUtilizationReportRequest()
    assert isinstance(response, future.Future)

def test_delete_utilization_report_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        client.delete_utilization_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteUtilizationReportRequest()

@pytest.mark.asyncio
async def test_delete_utilization_report_async(transport: str='grpc_asyncio', request_type=vmmigration.DeleteUtilizationReportRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_utilization_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteUtilizationReportRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_utilization_report_async_from_dict():
    await test_delete_utilization_report_async(request_type=dict)

def test_delete_utilization_report_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteUtilizationReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_utilization_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_utilization_report_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteUtilizationReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_utilization_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_utilization_report_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_utilization_report(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_utilization_report_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_utilization_report(vmmigration.DeleteUtilizationReportRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_utilization_report_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_utilization_report), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_utilization_report(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_utilization_report_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_utilization_report(vmmigration.DeleteUtilizationReportRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListDatacenterConnectorsRequest, dict])
def test_list_datacenter_connectors(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.return_value = vmmigration.ListDatacenterConnectorsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_datacenter_connectors(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListDatacenterConnectorsRequest()
    assert isinstance(response, pagers.ListDatacenterConnectorsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_datacenter_connectors_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        client.list_datacenter_connectors()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListDatacenterConnectorsRequest()

@pytest.mark.asyncio
async def test_list_datacenter_connectors_async(transport: str='grpc_asyncio', request_type=vmmigration.ListDatacenterConnectorsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListDatacenterConnectorsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_datacenter_connectors(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListDatacenterConnectorsRequest()
    assert isinstance(response, pagers.ListDatacenterConnectorsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_datacenter_connectors_async_from_dict():
    await test_list_datacenter_connectors_async(request_type=dict)

def test_list_datacenter_connectors_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListDatacenterConnectorsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.return_value = vmmigration.ListDatacenterConnectorsResponse()
        client.list_datacenter_connectors(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_datacenter_connectors_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListDatacenterConnectorsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListDatacenterConnectorsResponse())
        await client.list_datacenter_connectors(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_datacenter_connectors_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.return_value = vmmigration.ListDatacenterConnectorsResponse()
        client.list_datacenter_connectors(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_datacenter_connectors_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_datacenter_connectors(vmmigration.ListDatacenterConnectorsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_datacenter_connectors_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.return_value = vmmigration.ListDatacenterConnectorsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListDatacenterConnectorsResponse())
        response = await client.list_datacenter_connectors(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_datacenter_connectors_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_datacenter_connectors(vmmigration.ListDatacenterConnectorsRequest(), parent='parent_value')

def test_list_datacenter_connectors_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.side_effect = (vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()], next_page_token='abc'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[], next_page_token='def'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector()], next_page_token='ghi'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_datacenter_connectors(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.DatacenterConnector) for i in results))

def test_list_datacenter_connectors_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__') as call:
        call.side_effect = (vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()], next_page_token='abc'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[], next_page_token='def'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector()], next_page_token='ghi'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()]), RuntimeError)
        pages = list(client.list_datacenter_connectors(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_datacenter_connectors_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()], next_page_token='abc'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[], next_page_token='def'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector()], next_page_token='ghi'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()]), RuntimeError)
        async_pager = await client.list_datacenter_connectors(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.DatacenterConnector) for i in responses))

@pytest.mark.asyncio
async def test_list_datacenter_connectors_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_datacenter_connectors), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()], next_page_token='abc'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[], next_page_token='def'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector()], next_page_token='ghi'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_datacenter_connectors(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetDatacenterConnectorRequest, dict])
def test_get_datacenter_connector(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        call.return_value = vmmigration.DatacenterConnector(name='name_value', registration_id='registration_id_value', service_account='service_account_value', version='version_value', bucket='bucket_value', state=vmmigration.DatacenterConnector.State.PENDING, appliance_infrastructure_version='appliance_infrastructure_version_value', appliance_software_version='appliance_software_version_value')
        response = client.get_datacenter_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetDatacenterConnectorRequest()
    assert isinstance(response, vmmigration.DatacenterConnector)
    assert response.name == 'name_value'
    assert response.registration_id == 'registration_id_value'
    assert response.service_account == 'service_account_value'
    assert response.version == 'version_value'
    assert response.bucket == 'bucket_value'
    assert response.state == vmmigration.DatacenterConnector.State.PENDING
    assert response.appliance_infrastructure_version == 'appliance_infrastructure_version_value'
    assert response.appliance_software_version == 'appliance_software_version_value'

def test_get_datacenter_connector_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        client.get_datacenter_connector()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetDatacenterConnectorRequest()

@pytest.mark.asyncio
async def test_get_datacenter_connector_async(transport: str='grpc_asyncio', request_type=vmmigration.GetDatacenterConnectorRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.DatacenterConnector(name='name_value', registration_id='registration_id_value', service_account='service_account_value', version='version_value', bucket='bucket_value', state=vmmigration.DatacenterConnector.State.PENDING, appliance_infrastructure_version='appliance_infrastructure_version_value', appliance_software_version='appliance_software_version_value'))
        response = await client.get_datacenter_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetDatacenterConnectorRequest()
    assert isinstance(response, vmmigration.DatacenterConnector)
    assert response.name == 'name_value'
    assert response.registration_id == 'registration_id_value'
    assert response.service_account == 'service_account_value'
    assert response.version == 'version_value'
    assert response.bucket == 'bucket_value'
    assert response.state == vmmigration.DatacenterConnector.State.PENDING
    assert response.appliance_infrastructure_version == 'appliance_infrastructure_version_value'
    assert response.appliance_software_version == 'appliance_software_version_value'

@pytest.mark.asyncio
async def test_get_datacenter_connector_async_from_dict():
    await test_get_datacenter_connector_async(request_type=dict)

def test_get_datacenter_connector_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetDatacenterConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        call.return_value = vmmigration.DatacenterConnector()
        client.get_datacenter_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_datacenter_connector_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetDatacenterConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.DatacenterConnector())
        await client.get_datacenter_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_datacenter_connector_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        call.return_value = vmmigration.DatacenterConnector()
        client.get_datacenter_connector(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_datacenter_connector_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_datacenter_connector(vmmigration.GetDatacenterConnectorRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_datacenter_connector_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_datacenter_connector), '__call__') as call:
        call.return_value = vmmigration.DatacenterConnector()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.DatacenterConnector())
        response = await client.get_datacenter_connector(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_datacenter_connector_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_datacenter_connector(vmmigration.GetDatacenterConnectorRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateDatacenterConnectorRequest, dict])
def test_create_datacenter_connector(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_datacenter_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateDatacenterConnectorRequest()
    assert isinstance(response, future.Future)

def test_create_datacenter_connector_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        client.create_datacenter_connector()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateDatacenterConnectorRequest()

@pytest.mark.asyncio
async def test_create_datacenter_connector_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateDatacenterConnectorRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_datacenter_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateDatacenterConnectorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_datacenter_connector_async_from_dict():
    await test_create_datacenter_connector_async(request_type=dict)

def test_create_datacenter_connector_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateDatacenterConnectorRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_datacenter_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_datacenter_connector_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateDatacenterConnectorRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_datacenter_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_datacenter_connector_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_datacenter_connector(parent='parent_value', datacenter_connector=vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751)), datacenter_connector_id='datacenter_connector_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].datacenter_connector
        mock_val = vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].datacenter_connector_id
        mock_val = 'datacenter_connector_id_value'
        assert arg == mock_val

def test_create_datacenter_connector_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_datacenter_connector(vmmigration.CreateDatacenterConnectorRequest(), parent='parent_value', datacenter_connector=vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751)), datacenter_connector_id='datacenter_connector_id_value')

@pytest.mark.asyncio
async def test_create_datacenter_connector_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_datacenter_connector(parent='parent_value', datacenter_connector=vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751)), datacenter_connector_id='datacenter_connector_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].datacenter_connector
        mock_val = vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].datacenter_connector_id
        mock_val = 'datacenter_connector_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_datacenter_connector_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_datacenter_connector(vmmigration.CreateDatacenterConnectorRequest(), parent='parent_value', datacenter_connector=vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751)), datacenter_connector_id='datacenter_connector_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteDatacenterConnectorRequest, dict])
def test_delete_datacenter_connector(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_datacenter_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteDatacenterConnectorRequest()
    assert isinstance(response, future.Future)

def test_delete_datacenter_connector_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        client.delete_datacenter_connector()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteDatacenterConnectorRequest()

@pytest.mark.asyncio
async def test_delete_datacenter_connector_async(transport: str='grpc_asyncio', request_type=vmmigration.DeleteDatacenterConnectorRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_datacenter_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteDatacenterConnectorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_datacenter_connector_async_from_dict():
    await test_delete_datacenter_connector_async(request_type=dict)

def test_delete_datacenter_connector_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteDatacenterConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_datacenter_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_datacenter_connector_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteDatacenterConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_datacenter_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_datacenter_connector_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_datacenter_connector(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_datacenter_connector_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_datacenter_connector(vmmigration.DeleteDatacenterConnectorRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_datacenter_connector_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_datacenter_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_datacenter_connector(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_datacenter_connector_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_datacenter_connector(vmmigration.DeleteDatacenterConnectorRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.UpgradeApplianceRequest, dict])
def test_upgrade_appliance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upgrade_appliance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.upgrade_appliance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpgradeApplianceRequest()
    assert isinstance(response, future.Future)

def test_upgrade_appliance_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.upgrade_appliance), '__call__') as call:
        client.upgrade_appliance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpgradeApplianceRequest()

@pytest.mark.asyncio
async def test_upgrade_appliance_async(transport: str='grpc_asyncio', request_type=vmmigration.UpgradeApplianceRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upgrade_appliance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.upgrade_appliance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpgradeApplianceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_upgrade_appliance_async_from_dict():
    await test_upgrade_appliance_async(request_type=dict)

def test_upgrade_appliance_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpgradeApplianceRequest()
    request.datacenter_connector = 'datacenter_connector_value'
    with mock.patch.object(type(client.transport.upgrade_appliance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.upgrade_appliance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'datacenter_connector=datacenter_connector_value') in kw['metadata']

@pytest.mark.asyncio
async def test_upgrade_appliance_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpgradeApplianceRequest()
    request.datacenter_connector = 'datacenter_connector_value'
    with mock.patch.object(type(client.transport.upgrade_appliance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.upgrade_appliance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'datacenter_connector=datacenter_connector_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [vmmigration.CreateMigratingVmRequest, dict])
def test_create_migrating_vm(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateMigratingVmRequest()
    assert isinstance(response, future.Future)

def test_create_migrating_vm_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        client.create_migrating_vm()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateMigratingVmRequest()

@pytest.mark.asyncio
async def test_create_migrating_vm_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateMigratingVmRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateMigratingVmRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_migrating_vm_async_from_dict():
    await test_create_migrating_vm_async(request_type=dict)

def test_create_migrating_vm_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateMigratingVmRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_migrating_vm_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateMigratingVmRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_migrating_vm_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_migrating_vm(parent='parent_value', migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), migrating_vm_id='migrating_vm_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].migrating_vm
        mock_val = vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].migrating_vm_id
        mock_val = 'migrating_vm_id_value'
        assert arg == mock_val

def test_create_migrating_vm_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_migrating_vm(vmmigration.CreateMigratingVmRequest(), parent='parent_value', migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), migrating_vm_id='migrating_vm_id_value')

@pytest.mark.asyncio
async def test_create_migrating_vm_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_migrating_vm(parent='parent_value', migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), migrating_vm_id='migrating_vm_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].migrating_vm
        mock_val = vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].migrating_vm_id
        mock_val = 'migrating_vm_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_migrating_vm_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_migrating_vm(vmmigration.CreateMigratingVmRequest(), parent='parent_value', migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), migrating_vm_id='migrating_vm_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListMigratingVmsRequest, dict])
def test_list_migrating_vms(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.return_value = vmmigration.ListMigratingVmsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_migrating_vms(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListMigratingVmsRequest()
    assert isinstance(response, pagers.ListMigratingVmsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_migrating_vms_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        client.list_migrating_vms()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListMigratingVmsRequest()

@pytest.mark.asyncio
async def test_list_migrating_vms_async(transport: str='grpc_asyncio', request_type=vmmigration.ListMigratingVmsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListMigratingVmsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_migrating_vms(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListMigratingVmsRequest()
    assert isinstance(response, pagers.ListMigratingVmsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_migrating_vms_async_from_dict():
    await test_list_migrating_vms_async(request_type=dict)

def test_list_migrating_vms_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListMigratingVmsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.return_value = vmmigration.ListMigratingVmsResponse()
        client.list_migrating_vms(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_migrating_vms_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListMigratingVmsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListMigratingVmsResponse())
        await client.list_migrating_vms(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_migrating_vms_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.return_value = vmmigration.ListMigratingVmsResponse()
        client.list_migrating_vms(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_migrating_vms_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_migrating_vms(vmmigration.ListMigratingVmsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_migrating_vms_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.return_value = vmmigration.ListMigratingVmsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListMigratingVmsResponse())
        response = await client.list_migrating_vms(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_migrating_vms_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_migrating_vms(vmmigration.ListMigratingVmsRequest(), parent='parent_value')

def test_list_migrating_vms_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.side_effect = (vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm(), vmmigration.MigratingVm()], next_page_token='abc'), vmmigration.ListMigratingVmsResponse(migrating_vms=[], next_page_token='def'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm()], next_page_token='ghi'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_migrating_vms(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.MigratingVm) for i in results))

def test_list_migrating_vms_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__') as call:
        call.side_effect = (vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm(), vmmigration.MigratingVm()], next_page_token='abc'), vmmigration.ListMigratingVmsResponse(migrating_vms=[], next_page_token='def'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm()], next_page_token='ghi'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm()]), RuntimeError)
        pages = list(client.list_migrating_vms(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_migrating_vms_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm(), vmmigration.MigratingVm()], next_page_token='abc'), vmmigration.ListMigratingVmsResponse(migrating_vms=[], next_page_token='def'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm()], next_page_token='ghi'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm()]), RuntimeError)
        async_pager = await client.list_migrating_vms(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.MigratingVm) for i in responses))

@pytest.mark.asyncio
async def test_list_migrating_vms_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_migrating_vms), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm(), vmmigration.MigratingVm()], next_page_token='abc'), vmmigration.ListMigratingVmsResponse(migrating_vms=[], next_page_token='def'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm()], next_page_token='ghi'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_migrating_vms(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetMigratingVmRequest, dict])
def test_get_migrating_vm(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        call.return_value = vmmigration.MigratingVm(name='name_value', source_vm_id='source_vm_id_value', display_name='display_name_value', description='description_value', state=vmmigration.MigratingVm.State.PENDING, group='group_value')
        response = client.get_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetMigratingVmRequest()
    assert isinstance(response, vmmigration.MigratingVm)
    assert response.name == 'name_value'
    assert response.source_vm_id == 'source_vm_id_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == vmmigration.MigratingVm.State.PENDING
    assert response.group == 'group_value'

def test_get_migrating_vm_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        client.get_migrating_vm()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetMigratingVmRequest()

@pytest.mark.asyncio
async def test_get_migrating_vm_async(transport: str='grpc_asyncio', request_type=vmmigration.GetMigratingVmRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.MigratingVm(name='name_value', source_vm_id='source_vm_id_value', display_name='display_name_value', description='description_value', state=vmmigration.MigratingVm.State.PENDING, group='group_value'))
        response = await client.get_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetMigratingVmRequest()
    assert isinstance(response, vmmigration.MigratingVm)
    assert response.name == 'name_value'
    assert response.source_vm_id == 'source_vm_id_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == vmmigration.MigratingVm.State.PENDING
    assert response.group == 'group_value'

@pytest.mark.asyncio
async def test_get_migrating_vm_async_from_dict():
    await test_get_migrating_vm_async(request_type=dict)

def test_get_migrating_vm_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetMigratingVmRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        call.return_value = vmmigration.MigratingVm()
        client.get_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_migrating_vm_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetMigratingVmRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.MigratingVm())
        await client.get_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_migrating_vm_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        call.return_value = vmmigration.MigratingVm()
        client.get_migrating_vm(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_migrating_vm_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_migrating_vm(vmmigration.GetMigratingVmRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_migrating_vm_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_migrating_vm), '__call__') as call:
        call.return_value = vmmigration.MigratingVm()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.MigratingVm())
        response = await client.get_migrating_vm(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_migrating_vm_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_migrating_vm(vmmigration.GetMigratingVmRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateMigratingVmRequest, dict])
def test_update_migrating_vm(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateMigratingVmRequest()
    assert isinstance(response, future.Future)

def test_update_migrating_vm_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        client.update_migrating_vm()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateMigratingVmRequest()

@pytest.mark.asyncio
async def test_update_migrating_vm_async(transport: str='grpc_asyncio', request_type=vmmigration.UpdateMigratingVmRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateMigratingVmRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_migrating_vm_async_from_dict():
    await test_update_migrating_vm_async(request_type=dict)

def test_update_migrating_vm_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateMigratingVmRequest()
    request.migrating_vm.name = 'name_value'
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_migrating_vm_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateMigratingVmRequest()
    request.migrating_vm.name = 'name_value'
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm.name=name_value') in kw['metadata']

def test_update_migrating_vm_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_migrating_vm(migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].migrating_vm
        mock_val = vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_migrating_vm_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_migrating_vm(vmmigration.UpdateMigratingVmRequest(), migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_migrating_vm_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_migrating_vm(migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].migrating_vm
        mock_val = vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_migrating_vm_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_migrating_vm(vmmigration.UpdateMigratingVmRequest(), migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmmigration.DeleteMigratingVmRequest, dict])
def test_delete_migrating_vm(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteMigratingVmRequest()
    assert isinstance(response, future.Future)

def test_delete_migrating_vm_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        client.delete_migrating_vm()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteMigratingVmRequest()

@pytest.mark.asyncio
async def test_delete_migrating_vm_async(transport: str='grpc_asyncio', request_type=vmmigration.DeleteMigratingVmRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteMigratingVmRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_migrating_vm_async_from_dict():
    await test_delete_migrating_vm_async(request_type=dict)

def test_delete_migrating_vm_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteMigratingVmRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_migrating_vm(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_migrating_vm_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteMigratingVmRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_migrating_vm(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_migrating_vm_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_migrating_vm(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_migrating_vm_flattened_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_migrating_vm(vmmigration.DeleteMigratingVmRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_migrating_vm_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_migrating_vm), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_migrating_vm(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_migrating_vm_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_migrating_vm(vmmigration.DeleteMigratingVmRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.StartMigrationRequest, dict])
def test_start_migration(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.StartMigrationRequest()
    assert isinstance(response, future.Future)

def test_start_migration_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        client.start_migration()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.StartMigrationRequest()

@pytest.mark.asyncio
async def test_start_migration_async(transport: str='grpc_asyncio', request_type=vmmigration.StartMigrationRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.StartMigrationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_migration_async_from_dict():
    await test_start_migration_async(request_type=dict)

def test_start_migration_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.StartMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_migration_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.StartMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

def test_start_migration_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_migration(migrating_vm='migrating_vm_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].migrating_vm
        mock_val = 'migrating_vm_value'
        assert arg == mock_val

def test_start_migration_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.start_migration(vmmigration.StartMigrationRequest(), migrating_vm='migrating_vm_value')

@pytest.mark.asyncio
async def test_start_migration_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_migration(migrating_vm='migrating_vm_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].migrating_vm
        mock_val = 'migrating_vm_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_start_migration_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.start_migration(vmmigration.StartMigrationRequest(), migrating_vm='migrating_vm_value')

@pytest.mark.parametrize('request_type', [vmmigration.ResumeMigrationRequest, dict])
def test_resume_migration(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.resume_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ResumeMigrationRequest()
    assert isinstance(response, future.Future)

def test_resume_migration_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_migration), '__call__') as call:
        client.resume_migration()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ResumeMigrationRequest()

@pytest.mark.asyncio
async def test_resume_migration_async(transport: str='grpc_asyncio', request_type=vmmigration.ResumeMigrationRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.resume_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ResumeMigrationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_resume_migration_async_from_dict():
    await test_resume_migration_async(request_type=dict)

def test_resume_migration_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ResumeMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.resume_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.resume_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_migration_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ResumeMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.resume_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.resume_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [vmmigration.PauseMigrationRequest, dict])
def test_pause_migration(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.pause_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.PauseMigrationRequest()
    assert isinstance(response, future.Future)

def test_pause_migration_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.pause_migration), '__call__') as call:
        client.pause_migration()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.PauseMigrationRequest()

@pytest.mark.asyncio
async def test_pause_migration_async(transport: str='grpc_asyncio', request_type=vmmigration.PauseMigrationRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.pause_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.PauseMigrationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_pause_migration_async_from_dict():
    await test_pause_migration_async(request_type=dict)

def test_pause_migration_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.PauseMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.pause_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.pause_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

@pytest.mark.asyncio
async def test_pause_migration_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.PauseMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.pause_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.pause_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [vmmigration.FinalizeMigrationRequest, dict])
def test_finalize_migration(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.finalize_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.FinalizeMigrationRequest()
    assert isinstance(response, future.Future)

def test_finalize_migration_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        client.finalize_migration()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.FinalizeMigrationRequest()

@pytest.mark.asyncio
async def test_finalize_migration_async(transport: str='grpc_asyncio', request_type=vmmigration.FinalizeMigrationRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.finalize_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.FinalizeMigrationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_finalize_migration_async_from_dict():
    await test_finalize_migration_async(request_type=dict)

def test_finalize_migration_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.FinalizeMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.finalize_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

@pytest.mark.asyncio
async def test_finalize_migration_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.FinalizeMigrationRequest()
    request.migrating_vm = 'migrating_vm_value'
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.finalize_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'migrating_vm=migrating_vm_value') in kw['metadata']

def test_finalize_migration_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.finalize_migration(migrating_vm='migrating_vm_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].migrating_vm
        mock_val = 'migrating_vm_value'
        assert arg == mock_val

def test_finalize_migration_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.finalize_migration(vmmigration.FinalizeMigrationRequest(), migrating_vm='migrating_vm_value')

@pytest.mark.asyncio
async def test_finalize_migration_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.finalize_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.finalize_migration(migrating_vm='migrating_vm_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].migrating_vm
        mock_val = 'migrating_vm_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_finalize_migration_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.finalize_migration(vmmigration.FinalizeMigrationRequest(), migrating_vm='migrating_vm_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateCloneJobRequest, dict])
def test_create_clone_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_clone_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateCloneJobRequest()
    assert isinstance(response, future.Future)

def test_create_clone_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        client.create_clone_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateCloneJobRequest()

@pytest.mark.asyncio
async def test_create_clone_job_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateCloneJobRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_clone_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateCloneJobRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_clone_job_async_from_dict():
    await test_create_clone_job_async(request_type=dict)

def test_create_clone_job_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateCloneJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_clone_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_clone_job_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateCloneJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_clone_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_clone_job_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_clone_job(parent='parent_value', clone_job=vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), clone_job_id='clone_job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].clone_job
        mock_val = vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].clone_job_id
        mock_val = 'clone_job_id_value'
        assert arg == mock_val

def test_create_clone_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_clone_job(vmmigration.CreateCloneJobRequest(), parent='parent_value', clone_job=vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), clone_job_id='clone_job_id_value')

@pytest.mark.asyncio
async def test_create_clone_job_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_clone_job(parent='parent_value', clone_job=vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), clone_job_id='clone_job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].clone_job
        mock_val = vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].clone_job_id
        mock_val = 'clone_job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_clone_job_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_clone_job(vmmigration.CreateCloneJobRequest(), parent='parent_value', clone_job=vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), clone_job_id='clone_job_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.CancelCloneJobRequest, dict])
def test_cancel_clone_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.cancel_clone_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CancelCloneJobRequest()
    assert isinstance(response, future.Future)

def test_cancel_clone_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        client.cancel_clone_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CancelCloneJobRequest()

@pytest.mark.asyncio
async def test_cancel_clone_job_async(transport: str='grpc_asyncio', request_type=vmmigration.CancelCloneJobRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.cancel_clone_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CancelCloneJobRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_cancel_clone_job_async_from_dict():
    await test_cancel_clone_job_async(request_type=dict)

def test_cancel_clone_job_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CancelCloneJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.cancel_clone_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_clone_job_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CancelCloneJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.cancel_clone_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_cancel_clone_job_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.cancel_clone_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_cancel_clone_job_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.cancel_clone_job(vmmigration.CancelCloneJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_cancel_clone_job_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_clone_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.cancel_clone_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_cancel_clone_job_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.cancel_clone_job(vmmigration.CancelCloneJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListCloneJobsRequest, dict])
def test_list_clone_jobs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCloneJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_clone_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListCloneJobsRequest()
    assert isinstance(response, pagers.ListCloneJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_clone_jobs_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        client.list_clone_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListCloneJobsRequest()

@pytest.mark.asyncio
async def test_list_clone_jobs_async(transport: str='grpc_asyncio', request_type=vmmigration.ListCloneJobsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListCloneJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_clone_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListCloneJobsRequest()
    assert isinstance(response, pagers.ListCloneJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_clone_jobs_async_from_dict():
    await test_list_clone_jobs_async(request_type=dict)

def test_list_clone_jobs_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListCloneJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCloneJobsResponse()
        client.list_clone_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_clone_jobs_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListCloneJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListCloneJobsResponse())
        await client.list_clone_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_clone_jobs_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCloneJobsResponse()
        client.list_clone_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_clone_jobs_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_clone_jobs(vmmigration.ListCloneJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_clone_jobs_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCloneJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListCloneJobsResponse())
        response = await client.list_clone_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_clone_jobs_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_clone_jobs(vmmigration.ListCloneJobsRequest(), parent='parent_value')

def test_list_clone_jobs_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.side_effect = (vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob(), vmmigration.CloneJob()], next_page_token='abc'), vmmigration.ListCloneJobsResponse(clone_jobs=[], next_page_token='def'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob()], next_page_token='ghi'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_clone_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.CloneJob) for i in results))

def test_list_clone_jobs_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__') as call:
        call.side_effect = (vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob(), vmmigration.CloneJob()], next_page_token='abc'), vmmigration.ListCloneJobsResponse(clone_jobs=[], next_page_token='def'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob()], next_page_token='ghi'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob()]), RuntimeError)
        pages = list(client.list_clone_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_clone_jobs_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob(), vmmigration.CloneJob()], next_page_token='abc'), vmmigration.ListCloneJobsResponse(clone_jobs=[], next_page_token='def'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob()], next_page_token='ghi'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob()]), RuntimeError)
        async_pager = await client.list_clone_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.CloneJob) for i in responses))

@pytest.mark.asyncio
async def test_list_clone_jobs_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_clone_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob(), vmmigration.CloneJob()], next_page_token='abc'), vmmigration.ListCloneJobsResponse(clone_jobs=[], next_page_token='def'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob()], next_page_token='ghi'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_clone_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetCloneJobRequest, dict])
def test_get_clone_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        call.return_value = vmmigration.CloneJob(name='name_value', state=vmmigration.CloneJob.State.PENDING)
        response = client.get_clone_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetCloneJobRequest()
    assert isinstance(response, vmmigration.CloneJob)
    assert response.name == 'name_value'
    assert response.state == vmmigration.CloneJob.State.PENDING

def test_get_clone_job_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        client.get_clone_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetCloneJobRequest()

@pytest.mark.asyncio
async def test_get_clone_job_async(transport: str='grpc_asyncio', request_type=vmmigration.GetCloneJobRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.CloneJob(name='name_value', state=vmmigration.CloneJob.State.PENDING))
        response = await client.get_clone_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetCloneJobRequest()
    assert isinstance(response, vmmigration.CloneJob)
    assert response.name == 'name_value'
    assert response.state == vmmigration.CloneJob.State.PENDING

@pytest.mark.asyncio
async def test_get_clone_job_async_from_dict():
    await test_get_clone_job_async(request_type=dict)

def test_get_clone_job_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetCloneJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        call.return_value = vmmigration.CloneJob()
        client.get_clone_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_clone_job_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetCloneJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.CloneJob())
        await client.get_clone_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_clone_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        call.return_value = vmmigration.CloneJob()
        client.get_clone_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_clone_job_flattened_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_clone_job(vmmigration.GetCloneJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_clone_job_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_clone_job), '__call__') as call:
        call.return_value = vmmigration.CloneJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.CloneJob())
        response = await client.get_clone_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_clone_job_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_clone_job(vmmigration.GetCloneJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateCutoverJobRequest, dict])
def test_create_cutover_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_cutover_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateCutoverJobRequest()
    assert isinstance(response, future.Future)

def test_create_cutover_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        client.create_cutover_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateCutoverJobRequest()

@pytest.mark.asyncio
async def test_create_cutover_job_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateCutoverJobRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cutover_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateCutoverJobRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_cutover_job_async_from_dict():
    await test_create_cutover_job_async(request_type=dict)

def test_create_cutover_job_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateCutoverJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cutover_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_cutover_job_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateCutoverJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_cutover_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_cutover_job_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cutover_job(parent='parent_value', cutover_job=vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), cutover_job_id='cutover_job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cutover_job
        mock_val = vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].cutover_job_id
        mock_val = 'cutover_job_id_value'
        assert arg == mock_val

def test_create_cutover_job_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_cutover_job(vmmigration.CreateCutoverJobRequest(), parent='parent_value', cutover_job=vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), cutover_job_id='cutover_job_id_value')

@pytest.mark.asyncio
async def test_create_cutover_job_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cutover_job(parent='parent_value', cutover_job=vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), cutover_job_id='cutover_job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cutover_job
        mock_val = vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value'))
        assert arg == mock_val
        arg = args[0].cutover_job_id
        mock_val = 'cutover_job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_cutover_job_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_cutover_job(vmmigration.CreateCutoverJobRequest(), parent='parent_value', cutover_job=vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), cutover_job_id='cutover_job_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.CancelCutoverJobRequest, dict])
def test_cancel_cutover_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.cancel_cutover_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CancelCutoverJobRequest()
    assert isinstance(response, future.Future)

def test_cancel_cutover_job_empty_call():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        client.cancel_cutover_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CancelCutoverJobRequest()

@pytest.mark.asyncio
async def test_cancel_cutover_job_async(transport: str='grpc_asyncio', request_type=vmmigration.CancelCutoverJobRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.cancel_cutover_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CancelCutoverJobRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_cancel_cutover_job_async_from_dict():
    await test_cancel_cutover_job_async(request_type=dict)

def test_cancel_cutover_job_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CancelCutoverJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.cancel_cutover_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_cutover_job_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CancelCutoverJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.cancel_cutover_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_cancel_cutover_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.cancel_cutover_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_cancel_cutover_job_flattened_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.cancel_cutover_job(vmmigration.CancelCutoverJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_cancel_cutover_job_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_cutover_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.cancel_cutover_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_cancel_cutover_job_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.cancel_cutover_job(vmmigration.CancelCutoverJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListCutoverJobsRequest, dict])
def test_list_cutover_jobs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCutoverJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_cutover_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListCutoverJobsRequest()
    assert isinstance(response, pagers.ListCutoverJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_cutover_jobs_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        client.list_cutover_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListCutoverJobsRequest()

@pytest.mark.asyncio
async def test_list_cutover_jobs_async(transport: str='grpc_asyncio', request_type=vmmigration.ListCutoverJobsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListCutoverJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_cutover_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListCutoverJobsRequest()
    assert isinstance(response, pagers.ListCutoverJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_cutover_jobs_async_from_dict():
    await test_list_cutover_jobs_async(request_type=dict)

def test_list_cutover_jobs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListCutoverJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCutoverJobsResponse()
        client.list_cutover_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_cutover_jobs_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListCutoverJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListCutoverJobsResponse())
        await client.list_cutover_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_cutover_jobs_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCutoverJobsResponse()
        client.list_cutover_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_cutover_jobs_flattened_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_cutover_jobs(vmmigration.ListCutoverJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_cutover_jobs_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.return_value = vmmigration.ListCutoverJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListCutoverJobsResponse())
        response = await client.list_cutover_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_cutover_jobs_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_cutover_jobs(vmmigration.ListCutoverJobsRequest(), parent='parent_value')

def test_list_cutover_jobs_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.side_effect = (vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob(), vmmigration.CutoverJob()], next_page_token='abc'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[], next_page_token='def'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob()], next_page_token='ghi'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_cutover_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.CutoverJob) for i in results))

def test_list_cutover_jobs_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__') as call:
        call.side_effect = (vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob(), vmmigration.CutoverJob()], next_page_token='abc'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[], next_page_token='def'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob()], next_page_token='ghi'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob()]), RuntimeError)
        pages = list(client.list_cutover_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_cutover_jobs_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob(), vmmigration.CutoverJob()], next_page_token='abc'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[], next_page_token='def'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob()], next_page_token='ghi'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob()]), RuntimeError)
        async_pager = await client.list_cutover_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.CutoverJob) for i in responses))

@pytest.mark.asyncio
async def test_list_cutover_jobs_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_cutover_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob(), vmmigration.CutoverJob()], next_page_token='abc'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[], next_page_token='def'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob()], next_page_token='ghi'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_cutover_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetCutoverJobRequest, dict])
def test_get_cutover_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        call.return_value = vmmigration.CutoverJob(name='name_value', state=vmmigration.CutoverJob.State.PENDING, progress_percent=1733, state_message='state_message_value')
        response = client.get_cutover_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetCutoverJobRequest()
    assert isinstance(response, vmmigration.CutoverJob)
    assert response.name == 'name_value'
    assert response.state == vmmigration.CutoverJob.State.PENDING
    assert response.progress_percent == 1733
    assert response.state_message == 'state_message_value'

def test_get_cutover_job_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        client.get_cutover_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetCutoverJobRequest()

@pytest.mark.asyncio
async def test_get_cutover_job_async(transport: str='grpc_asyncio', request_type=vmmigration.GetCutoverJobRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.CutoverJob(name='name_value', state=vmmigration.CutoverJob.State.PENDING, progress_percent=1733, state_message='state_message_value'))
        response = await client.get_cutover_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetCutoverJobRequest()
    assert isinstance(response, vmmigration.CutoverJob)
    assert response.name == 'name_value'
    assert response.state == vmmigration.CutoverJob.State.PENDING
    assert response.progress_percent == 1733
    assert response.state_message == 'state_message_value'

@pytest.mark.asyncio
async def test_get_cutover_job_async_from_dict():
    await test_get_cutover_job_async(request_type=dict)

def test_get_cutover_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetCutoverJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        call.return_value = vmmigration.CutoverJob()
        client.get_cutover_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_cutover_job_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetCutoverJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.CutoverJob())
        await client.get_cutover_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_cutover_job_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        call.return_value = vmmigration.CutoverJob()
        client.get_cutover_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_cutover_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_cutover_job(vmmigration.GetCutoverJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_cutover_job_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cutover_job), '__call__') as call:
        call.return_value = vmmigration.CutoverJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.CutoverJob())
        response = await client.get_cutover_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_cutover_job_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_cutover_job(vmmigration.GetCutoverJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListGroupsRequest, dict])
def test_list_groups(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = vmmigration.ListGroupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListGroupsRequest()
    assert isinstance(response, pagers.ListGroupsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_groups_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        client.list_groups()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListGroupsRequest()

@pytest.mark.asyncio
async def test_list_groups_async(transport: str='grpc_asyncio', request_type=vmmigration.ListGroupsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListGroupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListGroupsRequest()
    assert isinstance(response, pagers.ListGroupsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_groups_async_from_dict():
    await test_list_groups_async(request_type=dict)

def test_list_groups_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = vmmigration.ListGroupsResponse()
        client.list_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_groups_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListGroupsResponse())
        await client.list_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_groups_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = vmmigration.ListGroupsResponse()
        client.list_groups(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_groups_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_groups(vmmigration.ListGroupsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_groups_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = vmmigration.ListGroupsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListGroupsResponse())
        response = await client.list_groups(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_groups_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_groups(vmmigration.ListGroupsRequest(), parent='parent_value')

def test_list_groups_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.side_effect = (vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group(), vmmigration.Group()], next_page_token='abc'), vmmigration.ListGroupsResponse(groups=[], next_page_token='def'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group()], next_page_token='ghi'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_groups(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.Group) for i in results))

def test_list_groups_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.side_effect = (vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group(), vmmigration.Group()], next_page_token='abc'), vmmigration.ListGroupsResponse(groups=[], next_page_token='def'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group()], next_page_token='ghi'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group()]), RuntimeError)
        pages = list(client.list_groups(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_groups_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group(), vmmigration.Group()], next_page_token='abc'), vmmigration.ListGroupsResponse(groups=[], next_page_token='def'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group()], next_page_token='ghi'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group()]), RuntimeError)
        async_pager = await client.list_groups(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.Group) for i in responses))

@pytest.mark.asyncio
async def test_list_groups_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group(), vmmigration.Group()], next_page_token='abc'), vmmigration.ListGroupsResponse(groups=[], next_page_token='def'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group()], next_page_token='ghi'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_groups(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetGroupRequest, dict])
def test_get_group(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = vmmigration.Group(name='name_value', description='description_value', display_name='display_name_value')
        response = client.get_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetGroupRequest()
    assert isinstance(response, vmmigration.Group)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'

def test_get_group_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        client.get_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetGroupRequest()

@pytest.mark.asyncio
async def test_get_group_async(transport: str='grpc_asyncio', request_type=vmmigration.GetGroupRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.Group(name='name_value', description='description_value', display_name='display_name_value'))
        response = await client.get_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetGroupRequest()
    assert isinstance(response, vmmigration.Group)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_group_async_from_dict():
    await test_get_group_async(request_type=dict)

def test_get_group_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = vmmigration.Group()
        client.get_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_group_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.Group())
        await client.get_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_group_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = vmmigration.Group()
        client.get_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_group_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_group(vmmigration.GetGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_group_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = vmmigration.Group()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.Group())
        response = await client.get_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_group_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_group(vmmigration.GetGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateGroupRequest, dict])
def test_create_group(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateGroupRequest()
    assert isinstance(response, future.Future)

def test_create_group_empty_call():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        client.create_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateGroupRequest()

@pytest.mark.asyncio
async def test_create_group_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateGroupRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateGroupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_group_async_from_dict():
    await test_create_group_async(request_type=dict)

def test_create_group_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_group_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_group_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_group(parent='parent_value', group=vmmigration.Group(name='name_value'), group_id='group_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].group
        mock_val = vmmigration.Group(name='name_value')
        assert arg == mock_val
        arg = args[0].group_id
        mock_val = 'group_id_value'
        assert arg == mock_val

def test_create_group_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_group(vmmigration.CreateGroupRequest(), parent='parent_value', group=vmmigration.Group(name='name_value'), group_id='group_id_value')

@pytest.mark.asyncio
async def test_create_group_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_group(parent='parent_value', group=vmmigration.Group(name='name_value'), group_id='group_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].group
        mock_val = vmmigration.Group(name='name_value')
        assert arg == mock_val
        arg = args[0].group_id
        mock_val = 'group_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_group_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_group(vmmigration.CreateGroupRequest(), parent='parent_value', group=vmmigration.Group(name='name_value'), group_id='group_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateGroupRequest, dict])
def test_update_group(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateGroupRequest()
    assert isinstance(response, future.Future)

def test_update_group_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        client.update_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateGroupRequest()

@pytest.mark.asyncio
async def test_update_group_async(transport: str='grpc_asyncio', request_type=vmmigration.UpdateGroupRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateGroupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_group_async_from_dict():
    await test_update_group_async(request_type=dict)

def test_update_group_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateGroupRequest()
    request.group.name = 'name_value'
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'group.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_group_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateGroupRequest()
    request.group.name = 'name_value'
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'group.name=name_value') in kw['metadata']

def test_update_group_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_group(group=vmmigration.Group(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].group
        mock_val = vmmigration.Group(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_group_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_group(vmmigration.UpdateGroupRequest(), group=vmmigration.Group(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_group_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_group(group=vmmigration.Group(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].group
        mock_val = vmmigration.Group(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_group_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_group(vmmigration.UpdateGroupRequest(), group=vmmigration.Group(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmmigration.DeleteGroupRequest, dict])
def test_delete_group(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteGroupRequest()
    assert isinstance(response, future.Future)

def test_delete_group_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        client.delete_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteGroupRequest()

@pytest.mark.asyncio
async def test_delete_group_async(transport: str='grpc_asyncio', request_type=vmmigration.DeleteGroupRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteGroupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_group_async_from_dict():
    await test_delete_group_async(request_type=dict)

def test_delete_group_field_headers():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_group_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_group_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_group_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_group(vmmigration.DeleteGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_group_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_group_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_group(vmmigration.DeleteGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.AddGroupMigrationRequest, dict])
def test_add_group_migration(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.add_group_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.AddGroupMigrationRequest()
    assert isinstance(response, future.Future)

def test_add_group_migration_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        client.add_group_migration()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.AddGroupMigrationRequest()

@pytest.mark.asyncio
async def test_add_group_migration_async(transport: str='grpc_asyncio', request_type=vmmigration.AddGroupMigrationRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_group_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.AddGroupMigrationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_add_group_migration_async_from_dict():
    await test_add_group_migration_async(request_type=dict)

def test_add_group_migration_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.AddGroupMigrationRequest()
    request.group = 'group_value'
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_group_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'group=group_value') in kw['metadata']

@pytest.mark.asyncio
async def test_add_group_migration_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.AddGroupMigrationRequest()
    request.group = 'group_value'
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.add_group_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'group=group_value') in kw['metadata']

def test_add_group_migration_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_group_migration(group='group_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].group
        mock_val = 'group_value'
        assert arg == mock_val

def test_add_group_migration_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.add_group_migration(vmmigration.AddGroupMigrationRequest(), group='group_value')

@pytest.mark.asyncio
async def test_add_group_migration_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_group_migration(group='group_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].group
        mock_val = 'group_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_add_group_migration_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.add_group_migration(vmmigration.AddGroupMigrationRequest(), group='group_value')

@pytest.mark.parametrize('request_type', [vmmigration.RemoveGroupMigrationRequest, dict])
def test_remove_group_migration(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.remove_group_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.RemoveGroupMigrationRequest()
    assert isinstance(response, future.Future)

def test_remove_group_migration_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        client.remove_group_migration()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.RemoveGroupMigrationRequest()

@pytest.mark.asyncio
async def test_remove_group_migration_async(transport: str='grpc_asyncio', request_type=vmmigration.RemoveGroupMigrationRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_group_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.RemoveGroupMigrationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_remove_group_migration_async_from_dict():
    await test_remove_group_migration_async(request_type=dict)

def test_remove_group_migration_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.RemoveGroupMigrationRequest()
    request.group = 'group_value'
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_group_migration(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'group=group_value') in kw['metadata']

@pytest.mark.asyncio
async def test_remove_group_migration_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.RemoveGroupMigrationRequest()
    request.group = 'group_value'
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.remove_group_migration(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'group=group_value') in kw['metadata']

def test_remove_group_migration_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_group_migration(group='group_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].group
        mock_val = 'group_value'
        assert arg == mock_val

def test_remove_group_migration_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.remove_group_migration(vmmigration.RemoveGroupMigrationRequest(), group='group_value')

@pytest.mark.asyncio
async def test_remove_group_migration_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.remove_group_migration), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_group_migration(group='group_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].group
        mock_val = 'group_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_remove_group_migration_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.remove_group_migration(vmmigration.RemoveGroupMigrationRequest(), group='group_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListTargetProjectsRequest, dict])
def test_list_target_projects(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.return_value = vmmigration.ListTargetProjectsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_target_projects(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListTargetProjectsRequest()
    assert isinstance(response, pagers.ListTargetProjectsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_target_projects_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        client.list_target_projects()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListTargetProjectsRequest()

@pytest.mark.asyncio
async def test_list_target_projects_async(transport: str='grpc_asyncio', request_type=vmmigration.ListTargetProjectsRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListTargetProjectsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_target_projects(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListTargetProjectsRequest()
    assert isinstance(response, pagers.ListTargetProjectsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_target_projects_async_from_dict():
    await test_list_target_projects_async(request_type=dict)

def test_list_target_projects_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListTargetProjectsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.return_value = vmmigration.ListTargetProjectsResponse()
        client.list_target_projects(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_target_projects_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListTargetProjectsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListTargetProjectsResponse())
        await client.list_target_projects(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_target_projects_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.return_value = vmmigration.ListTargetProjectsResponse()
        client.list_target_projects(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_target_projects_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_target_projects(vmmigration.ListTargetProjectsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_target_projects_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.return_value = vmmigration.ListTargetProjectsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListTargetProjectsResponse())
        response = await client.list_target_projects(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_target_projects_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_target_projects(vmmigration.ListTargetProjectsRequest(), parent='parent_value')

def test_list_target_projects_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.side_effect = (vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject(), vmmigration.TargetProject()], next_page_token='abc'), vmmigration.ListTargetProjectsResponse(target_projects=[], next_page_token='def'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject()], next_page_token='ghi'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_target_projects(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.TargetProject) for i in results))

def test_list_target_projects_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_target_projects), '__call__') as call:
        call.side_effect = (vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject(), vmmigration.TargetProject()], next_page_token='abc'), vmmigration.ListTargetProjectsResponse(target_projects=[], next_page_token='def'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject()], next_page_token='ghi'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject()]), RuntimeError)
        pages = list(client.list_target_projects(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_target_projects_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_target_projects), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject(), vmmigration.TargetProject()], next_page_token='abc'), vmmigration.ListTargetProjectsResponse(target_projects=[], next_page_token='def'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject()], next_page_token='ghi'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject()]), RuntimeError)
        async_pager = await client.list_target_projects(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.TargetProject) for i in responses))

@pytest.mark.asyncio
async def test_list_target_projects_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_target_projects), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject(), vmmigration.TargetProject()], next_page_token='abc'), vmmigration.ListTargetProjectsResponse(target_projects=[], next_page_token='def'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject()], next_page_token='ghi'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_target_projects(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetTargetProjectRequest, dict])
def test_get_target_project(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        call.return_value = vmmigration.TargetProject(name='name_value', project='project_value', description='description_value')
        response = client.get_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetTargetProjectRequest()
    assert isinstance(response, vmmigration.TargetProject)
    assert response.name == 'name_value'
    assert response.project == 'project_value'
    assert response.description == 'description_value'

def test_get_target_project_empty_call():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        client.get_target_project()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetTargetProjectRequest()

@pytest.mark.asyncio
async def test_get_target_project_async(transport: str='grpc_asyncio', request_type=vmmigration.GetTargetProjectRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.TargetProject(name='name_value', project='project_value', description='description_value'))
        response = await client.get_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetTargetProjectRequest()
    assert isinstance(response, vmmigration.TargetProject)
    assert response.name == 'name_value'
    assert response.project == 'project_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_get_target_project_async_from_dict():
    await test_get_target_project_async(request_type=dict)

def test_get_target_project_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetTargetProjectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        call.return_value = vmmigration.TargetProject()
        client.get_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_target_project_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetTargetProjectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.TargetProject())
        await client.get_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_target_project_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        call.return_value = vmmigration.TargetProject()
        client.get_target_project(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_target_project_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_target_project(vmmigration.GetTargetProjectRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_target_project_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_target_project), '__call__') as call:
        call.return_value = vmmigration.TargetProject()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.TargetProject())
        response = await client.get_target_project(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_target_project_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_target_project(vmmigration.GetTargetProjectRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.CreateTargetProjectRequest, dict])
def test_create_target_project(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateTargetProjectRequest()
    assert isinstance(response, future.Future)

def test_create_target_project_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        client.create_target_project()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateTargetProjectRequest()

@pytest.mark.asyncio
async def test_create_target_project_async(transport: str='grpc_asyncio', request_type=vmmigration.CreateTargetProjectRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.CreateTargetProjectRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_target_project_async_from_dict():
    await test_create_target_project_async(request_type=dict)

def test_create_target_project_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateTargetProjectRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_target_project_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.CreateTargetProjectRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_target_project_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_target_project(parent='parent_value', target_project=vmmigration.TargetProject(name='name_value'), target_project_id='target_project_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].target_project
        mock_val = vmmigration.TargetProject(name='name_value')
        assert arg == mock_val
        arg = args[0].target_project_id
        mock_val = 'target_project_id_value'
        assert arg == mock_val

def test_create_target_project_flattened_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_target_project(vmmigration.CreateTargetProjectRequest(), parent='parent_value', target_project=vmmigration.TargetProject(name='name_value'), target_project_id='target_project_id_value')

@pytest.mark.asyncio
async def test_create_target_project_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_target_project(parent='parent_value', target_project=vmmigration.TargetProject(name='name_value'), target_project_id='target_project_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].target_project
        mock_val = vmmigration.TargetProject(name='name_value')
        assert arg == mock_val
        arg = args[0].target_project_id
        mock_val = 'target_project_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_target_project_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_target_project(vmmigration.CreateTargetProjectRequest(), parent='parent_value', target_project=vmmigration.TargetProject(name='name_value'), target_project_id='target_project_id_value')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateTargetProjectRequest, dict])
def test_update_target_project(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateTargetProjectRequest()
    assert isinstance(response, future.Future)

def test_update_target_project_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        client.update_target_project()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateTargetProjectRequest()

@pytest.mark.asyncio
async def test_update_target_project_async(transport: str='grpc_asyncio', request_type=vmmigration.UpdateTargetProjectRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.UpdateTargetProjectRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_target_project_async_from_dict():
    await test_update_target_project_async(request_type=dict)

def test_update_target_project_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateTargetProjectRequest()
    request.target_project.name = 'name_value'
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'target_project.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_target_project_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.UpdateTargetProjectRequest()
    request.target_project.name = 'name_value'
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'target_project.name=name_value') in kw['metadata']

def test_update_target_project_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_target_project(target_project=vmmigration.TargetProject(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].target_project
        mock_val = vmmigration.TargetProject(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_target_project_flattened_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_target_project(vmmigration.UpdateTargetProjectRequest(), target_project=vmmigration.TargetProject(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_target_project_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_target_project(target_project=vmmigration.TargetProject(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].target_project
        mock_val = vmmigration.TargetProject(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_target_project_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_target_project(vmmigration.UpdateTargetProjectRequest(), target_project=vmmigration.TargetProject(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmmigration.DeleteTargetProjectRequest, dict])
def test_delete_target_project(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteTargetProjectRequest()
    assert isinstance(response, future.Future)

def test_delete_target_project_empty_call():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        client.delete_target_project()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteTargetProjectRequest()

@pytest.mark.asyncio
async def test_delete_target_project_async(transport: str='grpc_asyncio', request_type=vmmigration.DeleteTargetProjectRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.DeleteTargetProjectRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_target_project_async_from_dict():
    await test_delete_target_project_async(request_type=dict)

def test_delete_target_project_field_headers():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteTargetProjectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_target_project(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_target_project_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.DeleteTargetProjectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_target_project(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_target_project_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_target_project(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_target_project_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_target_project(vmmigration.DeleteTargetProjectRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_target_project_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_target_project), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_target_project(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_target_project_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_target_project(vmmigration.DeleteTargetProjectRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListReplicationCyclesRequest, dict])
def test_list_replication_cycles(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.return_value = vmmigration.ListReplicationCyclesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_replication_cycles(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListReplicationCyclesRequest()
    assert isinstance(response, pagers.ListReplicationCyclesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_replication_cycles_empty_call():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        client.list_replication_cycles()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListReplicationCyclesRequest()

@pytest.mark.asyncio
async def test_list_replication_cycles_async(transport: str='grpc_asyncio', request_type=vmmigration.ListReplicationCyclesRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListReplicationCyclesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_replication_cycles(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.ListReplicationCyclesRequest()
    assert isinstance(response, pagers.ListReplicationCyclesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_replication_cycles_async_from_dict():
    await test_list_replication_cycles_async(request_type=dict)

def test_list_replication_cycles_field_headers():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListReplicationCyclesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.return_value = vmmigration.ListReplicationCyclesResponse()
        client.list_replication_cycles(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_replication_cycles_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.ListReplicationCyclesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListReplicationCyclesResponse())
        await client.list_replication_cycles(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_replication_cycles_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.return_value = vmmigration.ListReplicationCyclesResponse()
        client.list_replication_cycles(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_replication_cycles_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_replication_cycles(vmmigration.ListReplicationCyclesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_replication_cycles_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.return_value = vmmigration.ListReplicationCyclesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ListReplicationCyclesResponse())
        response = await client.list_replication_cycles(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_replication_cycles_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_replication_cycles(vmmigration.ListReplicationCyclesRequest(), parent='parent_value')

def test_list_replication_cycles_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.side_effect = (vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()], next_page_token='abc'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[], next_page_token='def'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle()], next_page_token='ghi'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_replication_cycles(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.ReplicationCycle) for i in results))

def test_list_replication_cycles_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__') as call:
        call.side_effect = (vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()], next_page_token='abc'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[], next_page_token='def'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle()], next_page_token='ghi'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()]), RuntimeError)
        pages = list(client.list_replication_cycles(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_replication_cycles_async_pager():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()], next_page_token='abc'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[], next_page_token='def'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle()], next_page_token='ghi'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()]), RuntimeError)
        async_pager = await client.list_replication_cycles(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmmigration.ReplicationCycle) for i in responses))

@pytest.mark.asyncio
async def test_list_replication_cycles_async_pages():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_replication_cycles), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()], next_page_token='abc'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[], next_page_token='def'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle()], next_page_token='ghi'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_replication_cycles(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetReplicationCycleRequest, dict])
def test_get_replication_cycle(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        call.return_value = vmmigration.ReplicationCycle(name='name_value', cycle_number=1272, progress_percent=1733, state=vmmigration.ReplicationCycle.State.RUNNING)
        response = client.get_replication_cycle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetReplicationCycleRequest()
    assert isinstance(response, vmmigration.ReplicationCycle)
    assert response.name == 'name_value'
    assert response.cycle_number == 1272
    assert response.progress_percent == 1733
    assert response.state == vmmigration.ReplicationCycle.State.RUNNING

def test_get_replication_cycle_empty_call():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        client.get_replication_cycle()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetReplicationCycleRequest()

@pytest.mark.asyncio
async def test_get_replication_cycle_async(transport: str='grpc_asyncio', request_type=vmmigration.GetReplicationCycleRequest):
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ReplicationCycle(name='name_value', cycle_number=1272, progress_percent=1733, state=vmmigration.ReplicationCycle.State.RUNNING))
        response = await client.get_replication_cycle(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmmigration.GetReplicationCycleRequest()
    assert isinstance(response, vmmigration.ReplicationCycle)
    assert response.name == 'name_value'
    assert response.cycle_number == 1272
    assert response.progress_percent == 1733
    assert response.state == vmmigration.ReplicationCycle.State.RUNNING

@pytest.mark.asyncio
async def test_get_replication_cycle_async_from_dict():
    await test_get_replication_cycle_async(request_type=dict)

def test_get_replication_cycle_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetReplicationCycleRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        call.return_value = vmmigration.ReplicationCycle()
        client.get_replication_cycle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_replication_cycle_field_headers_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmmigration.GetReplicationCycleRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ReplicationCycle())
        await client.get_replication_cycle(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_replication_cycle_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        call.return_value = vmmigration.ReplicationCycle()
        client.get_replication_cycle(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_replication_cycle_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_replication_cycle(vmmigration.GetReplicationCycleRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_replication_cycle_flattened_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_replication_cycle), '__call__') as call:
        call.return_value = vmmigration.ReplicationCycle()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmmigration.ReplicationCycle())
        response = await client.get_replication_cycle(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_replication_cycle_flattened_error_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_replication_cycle(vmmigration.GetReplicationCycleRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmmigration.ListSourcesRequest, dict])
def test_list_sources_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListSourcesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListSourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_sources(request)
    assert isinstance(response, pagers.ListSourcesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_sources_rest_required_fields(request_type=vmmigration.ListSourcesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_sources._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_sources._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListSourcesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListSourcesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_sources(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_sources_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_sources._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_sources_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_sources') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_sources') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListSourcesRequest.pb(vmmigration.ListSourcesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListSourcesResponse.to_json(vmmigration.ListSourcesResponse())
        request = vmmigration.ListSourcesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListSourcesResponse()
        client.list_sources(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_sources_rest_bad_request(transport: str='rest', request_type=vmmigration.ListSourcesRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_sources(request)

def test_list_sources_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListSourcesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListSourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_sources(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/sources' % client.transport._host, args[1])

def test_list_sources_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_sources(vmmigration.ListSourcesRequest(), parent='parent_value')

def test_list_sources_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source(), vmmigration.Source()], next_page_token='abc'), vmmigration.ListSourcesResponse(sources=[], next_page_token='def'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source()], next_page_token='ghi'), vmmigration.ListSourcesResponse(sources=[vmmigration.Source(), vmmigration.Source()]))
        response = response + response
        response = tuple((vmmigration.ListSourcesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_sources(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.Source) for i in results))
        pages = list(client.list_sources(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetSourceRequest, dict])
def test_get_source_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.Source(name='name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.Source.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_source(request)
    assert isinstance(response, vmmigration.Source)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

def test_get_source_rest_required_fields(request_type=vmmigration.GetSourceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.Source()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.Source.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_source(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_source_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_source._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_source_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_source') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_source') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetSourceRequest.pb(vmmigration.GetSourceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.Source.to_json(vmmigration.Source())
        request = vmmigration.GetSourceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.Source()
        client.get_source(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_source_rest_bad_request(transport: str='rest', request_type=vmmigration.GetSourceRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_source(request)

def test_get_source_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.Source()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.Source.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_source(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*}' % client.transport._host, args[1])

def test_get_source_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_source(vmmigration.GetSourceRequest(), name='name_value')

def test_get_source_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateSourceRequest, dict])
def test_create_source_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['source'] = {'vmware': {'username': 'username_value', 'password': 'password_value', 'vcenter_ip': 'vcenter_ip_value', 'thumbprint': 'thumbprint_value'}, 'aws': {'access_key_creds': {'access_key_id': 'access_key_id_value', 'secret_access_key': 'secret_access_key_value'}, 'aws_region': 'aws_region_value', 'state': 1, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'inventory_tag_list': [{'key': 'key_value', 'value': 'value_value'}], 'inventory_security_group_names': ['inventory_security_group_names_value1', 'inventory_security_group_names_value2'], 'migration_resources_user_tags': {}, 'public_ip': 'public_ip_value'}, 'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value'}
    test_field = vmmigration.CreateSourceRequest.meta.fields['source']

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
    for (field, value) in request_init['source'].items():
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
                for i in range(0, len(request_init['source'][field])):
                    del request_init['source'][field][i][subfield]
            else:
                del request_init['source'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_source(request)
    assert response.operation.name == 'operations/spam'

def test_create_source_rest_required_fields(request_type=vmmigration.CreateSourceRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['source_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'sourceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'sourceId' in jsonified_request
    assert jsonified_request['sourceId'] == request_init['source_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['sourceId'] = 'source_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_source._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'source_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'sourceId' in jsonified_request
    assert jsonified_request['sourceId'] == 'source_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_source(request)
            expected_params = [('sourceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_source_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_source._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'sourceId')) & set(('parent', 'sourceId', 'source'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_source_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_source') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_source') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateSourceRequest.pb(vmmigration.CreateSourceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateSourceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_source(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_source_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateSourceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_source(request)

def test_create_source_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), source_id='source_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_source(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/sources' % client.transport._host, args[1])

def test_create_source_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_source(vmmigration.CreateSourceRequest(), parent='parent_value', source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), source_id='source_id_value')

def test_create_source_rest_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateSourceRequest, dict])
def test_update_source_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'source': {'name': 'projects/sample1/locations/sample2/sources/sample3'}}
    request_init['source'] = {'vmware': {'username': 'username_value', 'password': 'password_value', 'vcenter_ip': 'vcenter_ip_value', 'thumbprint': 'thumbprint_value'}, 'aws': {'access_key_creds': {'access_key_id': 'access_key_id_value', 'secret_access_key': 'secret_access_key_value'}, 'aws_region': 'aws_region_value', 'state': 1, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'inventory_tag_list': [{'key': 'key_value', 'value': 'value_value'}], 'inventory_security_group_names': ['inventory_security_group_names_value1', 'inventory_security_group_names_value2'], 'migration_resources_user_tags': {}, 'public_ip': 'public_ip_value'}, 'name': 'projects/sample1/locations/sample2/sources/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value'}
    test_field = vmmigration.UpdateSourceRequest.meta.fields['source']

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
    for (field, value) in request_init['source'].items():
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
                for i in range(0, len(request_init['source'][field])):
                    del request_init['source'][field][i][subfield]
            else:
                del request_init['source'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_source(request)
    assert response.operation.name == 'operations/spam'

def test_update_source_rest_required_fields(request_type=vmmigration.UpdateSourceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_source._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_source(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_source_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_source._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('source',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_source_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_update_source') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_update_source') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.UpdateSourceRequest.pb(vmmigration.UpdateSourceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.UpdateSourceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_source(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_source_rest_bad_request(transport: str='rest', request_type=vmmigration.UpdateSourceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'source': {'name': 'projects/sample1/locations/sample2/sources/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_source(request)

def test_update_source_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'source': {'name': 'projects/sample1/locations/sample2/sources/sample3'}}
        mock_args = dict(source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_source(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{source.name=projects/*/locations/*/sources/*}' % client.transport._host, args[1])

def test_update_source_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_source(vmmigration.UpdateSourceRequest(), source=vmmigration.Source(vmware=vmmigration.VmwareSourceDetails(username='username_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_source_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteSourceRequest, dict])
def test_delete_source_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_source(request)
    assert response.operation.name == 'operations/spam'

def test_delete_source_rest_required_fields(request_type=vmmigration.DeleteSourceRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_source._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_source(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_source_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_source._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_source_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_delete_source') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_delete_source') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.DeleteSourceRequest.pb(vmmigration.DeleteSourceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.DeleteSourceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_source(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_source_rest_bad_request(transport: str='rest', request_type=vmmigration.DeleteSourceRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_source(request)

def test_delete_source_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_source(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*}' % client.transport._host, args[1])

def test_delete_source_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_source(vmmigration.DeleteSourceRequest(), name='name_value')

def test_delete_source_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.FetchInventoryRequest, dict])
def test_fetch_inventory_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'source': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.FetchInventoryResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.FetchInventoryResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_inventory(request)
    assert isinstance(response, vmmigration.FetchInventoryResponse)

def test_fetch_inventory_rest_required_fields(request_type=vmmigration.FetchInventoryRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['source'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['source'] = 'source_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_inventory._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force_refresh',))
    jsonified_request.update(unset_fields)
    assert 'source' in jsonified_request
    assert jsonified_request['source'] == 'source_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.FetchInventoryResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.FetchInventoryResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_inventory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_inventory_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_inventory._get_unset_required_fields({})
    assert set(unset_fields) == set(('forceRefresh',)) & set(('source',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_inventory_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_fetch_inventory') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_fetch_inventory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.FetchInventoryRequest.pb(vmmigration.FetchInventoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.FetchInventoryResponse.to_json(vmmigration.FetchInventoryResponse())
        request = vmmigration.FetchInventoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.FetchInventoryResponse()
        client.fetch_inventory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_inventory_rest_bad_request(transport: str='rest', request_type=vmmigration.FetchInventoryRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'source': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_inventory(request)

def test_fetch_inventory_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.FetchInventoryResponse()
        sample_request = {'source': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(source='source_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.FetchInventoryResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.fetch_inventory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{source=projects/*/locations/*/sources/*}:fetchInventory' % client.transport._host, args[1])

def test_fetch_inventory_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.fetch_inventory(vmmigration.FetchInventoryRequest(), source='source_value')

def test_fetch_inventory_rest_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListUtilizationReportsRequest, dict])
def test_list_utilization_reports_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListUtilizationReportsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListUtilizationReportsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_utilization_reports(request)
    assert isinstance(response, pagers.ListUtilizationReportsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_utilization_reports_rest_required_fields(request_type=vmmigration.ListUtilizationReportsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_utilization_reports._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_utilization_reports._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListUtilizationReportsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListUtilizationReportsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_utilization_reports(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_utilization_reports_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_utilization_reports._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken', 'view')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_utilization_reports_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_utilization_reports') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_utilization_reports') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListUtilizationReportsRequest.pb(vmmigration.ListUtilizationReportsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListUtilizationReportsResponse.to_json(vmmigration.ListUtilizationReportsResponse())
        request = vmmigration.ListUtilizationReportsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListUtilizationReportsResponse()
        client.list_utilization_reports(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_utilization_reports_rest_bad_request(transport: str='rest', request_type=vmmigration.ListUtilizationReportsRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_utilization_reports(request)

def test_list_utilization_reports_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListUtilizationReportsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListUtilizationReportsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_utilization_reports(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*}/utilizationReports' % client.transport._host, args[1])

def test_list_utilization_reports_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_utilization_reports(vmmigration.ListUtilizationReportsRequest(), parent='parent_value')

def test_list_utilization_reports_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport(), vmmigration.UtilizationReport()], next_page_token='abc'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[], next_page_token='def'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport()], next_page_token='ghi'), vmmigration.ListUtilizationReportsResponse(utilization_reports=[vmmigration.UtilizationReport(), vmmigration.UtilizationReport()]))
        response = response + response
        response = tuple((vmmigration.ListUtilizationReportsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        pager = client.list_utilization_reports(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.UtilizationReport) for i in results))
        pages = list(client.list_utilization_reports(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetUtilizationReportRequest, dict])
def test_get_utilization_report_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/utilizationReports/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.UtilizationReport(name='name_value', display_name='display_name_value', state=vmmigration.UtilizationReport.State.CREATING, time_frame=vmmigration.UtilizationReport.TimeFrame.WEEK, vm_count=875)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.UtilizationReport.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_utilization_report(request)
    assert isinstance(response, vmmigration.UtilizationReport)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == vmmigration.UtilizationReport.State.CREATING
    assert response.time_frame == vmmigration.UtilizationReport.TimeFrame.WEEK
    assert response.vm_count == 875

def test_get_utilization_report_rest_required_fields(request_type=vmmigration.GetUtilizationReportRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_utilization_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_utilization_report._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.UtilizationReport()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.UtilizationReport.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_utilization_report(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_utilization_report_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_utilization_report._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_utilization_report_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_utilization_report') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_utilization_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetUtilizationReportRequest.pb(vmmigration.GetUtilizationReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.UtilizationReport.to_json(vmmigration.UtilizationReport())
        request = vmmigration.GetUtilizationReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.UtilizationReport()
        client.get_utilization_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_utilization_report_rest_bad_request(transport: str='rest', request_type=vmmigration.GetUtilizationReportRequest):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/utilizationReports/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_utilization_report(request)

def test_get_utilization_report_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.UtilizationReport()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/utilizationReports/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.UtilizationReport.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_utilization_report(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/utilizationReports/*}' % client.transport._host, args[1])

def test_get_utilization_report_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_utilization_report(vmmigration.GetUtilizationReportRequest(), name='name_value')

def test_get_utilization_report_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateUtilizationReportRequest, dict])
def test_create_utilization_report_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request_init['utilization_report'] = {'name': 'name_value', 'display_name': 'display_name_value', 'state': 1, 'state_time': {'seconds': 751, 'nanos': 543}, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'create_time': {}, 'time_frame': 1, 'frame_end_time': {}, 'vm_count': 875, 'vms': [{'vmware_vm_details': {'vm_id': 'vm_id_value', 'datacenter_id': 'datacenter_id_value', 'datacenter_description': 'datacenter_description_value', 'uuid': 'uuid_value', 'display_name': 'display_name_value', 'power_state': 1, 'cpu_count': 976, 'memory_mb': 967, 'disk_count': 1075, 'committed_storage_mb': 2120, 'guest_description': 'guest_description_value', 'boot_option': 1}, 'vm_id': 'vm_id_value', 'utilization': {'cpu_max_percent': 1597, 'cpu_average_percent': 2002, 'memory_max_percent': 1934, 'memory_average_percent': 2339, 'disk_io_rate_max_kbps': 2209, 'disk_io_rate_average_kbps': 2614, 'network_throughput_max_kbps': 2935, 'network_throughput_average_kbps': 3340}}]}
    test_field = vmmigration.CreateUtilizationReportRequest.meta.fields['utilization_report']

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
    for (field, value) in request_init['utilization_report'].items():
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
                for i in range(0, len(request_init['utilization_report'][field])):
                    del request_init['utilization_report'][field][i][subfield]
            else:
                del request_init['utilization_report'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_utilization_report(request)
    assert response.operation.name == 'operations/spam'

def test_create_utilization_report_rest_required_fields(request_type=vmmigration.CreateUtilizationReportRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['utilization_report_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'utilizationReportId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_utilization_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'utilizationReportId' in jsonified_request
    assert jsonified_request['utilizationReportId'] == request_init['utilization_report_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['utilizationReportId'] = 'utilization_report_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_utilization_report._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'utilization_report_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'utilizationReportId' in jsonified_request
    assert jsonified_request['utilizationReportId'] == 'utilization_report_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_utilization_report(request)
            expected_params = [('utilizationReportId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_utilization_report_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_utilization_report._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'utilizationReportId')) & set(('parent', 'utilizationReport', 'utilizationReportId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_utilization_report_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_utilization_report') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_utilization_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateUtilizationReportRequest.pb(vmmigration.CreateUtilizationReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateUtilizationReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_utilization_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_utilization_report_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateUtilizationReportRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_utilization_report(request)

def test_create_utilization_report_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(parent='parent_value', utilization_report=vmmigration.UtilizationReport(name='name_value'), utilization_report_id='utilization_report_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_utilization_report(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*}/utilizationReports' % client.transport._host, args[1])

def test_create_utilization_report_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_utilization_report(vmmigration.CreateUtilizationReportRequest(), parent='parent_value', utilization_report=vmmigration.UtilizationReport(name='name_value'), utilization_report_id='utilization_report_id_value')

def test_create_utilization_report_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteUtilizationReportRequest, dict])
def test_delete_utilization_report_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/utilizationReports/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_utilization_report(request)
    assert response.operation.name == 'operations/spam'

def test_delete_utilization_report_rest_required_fields(request_type=vmmigration.DeleteUtilizationReportRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_utilization_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_utilization_report._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_utilization_report(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_utilization_report_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_utilization_report._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_utilization_report_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_delete_utilization_report') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_delete_utilization_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.DeleteUtilizationReportRequest.pb(vmmigration.DeleteUtilizationReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.DeleteUtilizationReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_utilization_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_utilization_report_rest_bad_request(transport: str='rest', request_type=vmmigration.DeleteUtilizationReportRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/utilizationReports/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_utilization_report(request)

def test_delete_utilization_report_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/utilizationReports/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_utilization_report(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/utilizationReports/*}' % client.transport._host, args[1])

def test_delete_utilization_report_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_utilization_report(vmmigration.DeleteUtilizationReportRequest(), name='name_value')

def test_delete_utilization_report_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListDatacenterConnectorsRequest, dict])
def test_list_datacenter_connectors_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListDatacenterConnectorsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListDatacenterConnectorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_datacenter_connectors(request)
    assert isinstance(response, pagers.ListDatacenterConnectorsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_datacenter_connectors_rest_required_fields(request_type=vmmigration.ListDatacenterConnectorsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_datacenter_connectors._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_datacenter_connectors._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListDatacenterConnectorsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListDatacenterConnectorsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_datacenter_connectors(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_datacenter_connectors_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_datacenter_connectors._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_datacenter_connectors_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_datacenter_connectors') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_datacenter_connectors') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListDatacenterConnectorsRequest.pb(vmmigration.ListDatacenterConnectorsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListDatacenterConnectorsResponse.to_json(vmmigration.ListDatacenterConnectorsResponse())
        request = vmmigration.ListDatacenterConnectorsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListDatacenterConnectorsResponse()
        client.list_datacenter_connectors(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_datacenter_connectors_rest_bad_request(transport: str='rest', request_type=vmmigration.ListDatacenterConnectorsRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_datacenter_connectors(request)

def test_list_datacenter_connectors_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListDatacenterConnectorsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListDatacenterConnectorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_datacenter_connectors(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*}/datacenterConnectors' % client.transport._host, args[1])

def test_list_datacenter_connectors_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_datacenter_connectors(vmmigration.ListDatacenterConnectorsRequest(), parent='parent_value')

def test_list_datacenter_connectors_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()], next_page_token='abc'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[], next_page_token='def'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector()], next_page_token='ghi'), vmmigration.ListDatacenterConnectorsResponse(datacenter_connectors=[vmmigration.DatacenterConnector(), vmmigration.DatacenterConnector()]))
        response = response + response
        response = tuple((vmmigration.ListDatacenterConnectorsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        pager = client.list_datacenter_connectors(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.DatacenterConnector) for i in results))
        pages = list(client.list_datacenter_connectors(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetDatacenterConnectorRequest, dict])
def test_get_datacenter_connector_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.DatacenterConnector(name='name_value', registration_id='registration_id_value', service_account='service_account_value', version='version_value', bucket='bucket_value', state=vmmigration.DatacenterConnector.State.PENDING, appliance_infrastructure_version='appliance_infrastructure_version_value', appliance_software_version='appliance_software_version_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.DatacenterConnector.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_datacenter_connector(request)
    assert isinstance(response, vmmigration.DatacenterConnector)
    assert response.name == 'name_value'
    assert response.registration_id == 'registration_id_value'
    assert response.service_account == 'service_account_value'
    assert response.version == 'version_value'
    assert response.bucket == 'bucket_value'
    assert response.state == vmmigration.DatacenterConnector.State.PENDING
    assert response.appliance_infrastructure_version == 'appliance_infrastructure_version_value'
    assert response.appliance_software_version == 'appliance_software_version_value'

def test_get_datacenter_connector_rest_required_fields(request_type=vmmigration.GetDatacenterConnectorRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_datacenter_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_datacenter_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.DatacenterConnector()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.DatacenterConnector.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_datacenter_connector(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_datacenter_connector_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_datacenter_connector._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_datacenter_connector_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_datacenter_connector') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_datacenter_connector') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetDatacenterConnectorRequest.pb(vmmigration.GetDatacenterConnectorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.DatacenterConnector.to_json(vmmigration.DatacenterConnector())
        request = vmmigration.GetDatacenterConnectorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.DatacenterConnector()
        client.get_datacenter_connector(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_datacenter_connector_rest_bad_request(transport: str='rest', request_type=vmmigration.GetDatacenterConnectorRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_datacenter_connector(request)

def test_get_datacenter_connector_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.DatacenterConnector()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.DatacenterConnector.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_datacenter_connector(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/datacenterConnectors/*}' % client.transport._host, args[1])

def test_get_datacenter_connector_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_datacenter_connector(vmmigration.GetDatacenterConnectorRequest(), name='name_value')

def test_get_datacenter_connector_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateDatacenterConnectorRequest, dict])
def test_create_datacenter_connector_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request_init['datacenter_connector'] = {'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'name': 'name_value', 'registration_id': 'registration_id_value', 'service_account': 'service_account_value', 'version': 'version_value', 'bucket': 'bucket_value', 'state': 1, 'state_time': {}, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'appliance_infrastructure_version': 'appliance_infrastructure_version_value', 'appliance_software_version': 'appliance_software_version_value', 'available_versions': {'new_deployable_appliance': {'version': 'version_value', 'uri': 'uri_value', 'critical': True, 'release_notes_uri': 'release_notes_uri_value'}, 'in_place_update': {}}, 'upgrade_status': {'version': 'version_value', 'state': 1, 'error': {}, 'start_time': {}, 'previous_version': 'previous_version_value'}}
    test_field = vmmigration.CreateDatacenterConnectorRequest.meta.fields['datacenter_connector']

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
    for (field, value) in request_init['datacenter_connector'].items():
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
                for i in range(0, len(request_init['datacenter_connector'][field])):
                    del request_init['datacenter_connector'][field][i][subfield]
            else:
                del request_init['datacenter_connector'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_datacenter_connector(request)
    assert response.operation.name == 'operations/spam'

def test_create_datacenter_connector_rest_required_fields(request_type=vmmigration.CreateDatacenterConnectorRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['datacenter_connector_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'datacenterConnectorId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_datacenter_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'datacenterConnectorId' in jsonified_request
    assert jsonified_request['datacenterConnectorId'] == request_init['datacenter_connector_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['datacenterConnectorId'] = 'datacenter_connector_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_datacenter_connector._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('datacenter_connector_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'datacenterConnectorId' in jsonified_request
    assert jsonified_request['datacenterConnectorId'] == 'datacenter_connector_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_datacenter_connector(request)
            expected_params = [('datacenterConnectorId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_datacenter_connector_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_datacenter_connector._get_unset_required_fields({})
    assert set(unset_fields) == set(('datacenterConnectorId', 'requestId')) & set(('parent', 'datacenterConnectorId', 'datacenterConnector'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_datacenter_connector_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_datacenter_connector') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_datacenter_connector') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateDatacenterConnectorRequest.pb(vmmigration.CreateDatacenterConnectorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateDatacenterConnectorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_datacenter_connector(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_datacenter_connector_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateDatacenterConnectorRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_datacenter_connector(request)

def test_create_datacenter_connector_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(parent='parent_value', datacenter_connector=vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751)), datacenter_connector_id='datacenter_connector_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_datacenter_connector(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*}/datacenterConnectors' % client.transport._host, args[1])

def test_create_datacenter_connector_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_datacenter_connector(vmmigration.CreateDatacenterConnectorRequest(), parent='parent_value', datacenter_connector=vmmigration.DatacenterConnector(create_time=timestamp_pb2.Timestamp(seconds=751)), datacenter_connector_id='datacenter_connector_id_value')

def test_create_datacenter_connector_rest_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteDatacenterConnectorRequest, dict])
def test_delete_datacenter_connector_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_datacenter_connector(request)
    assert response.operation.name == 'operations/spam'

def test_delete_datacenter_connector_rest_required_fields(request_type=vmmigration.DeleteDatacenterConnectorRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_datacenter_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_datacenter_connector._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_datacenter_connector(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_datacenter_connector_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_datacenter_connector._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_datacenter_connector_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_delete_datacenter_connector') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_delete_datacenter_connector') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.DeleteDatacenterConnectorRequest.pb(vmmigration.DeleteDatacenterConnectorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.DeleteDatacenterConnectorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_datacenter_connector(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_datacenter_connector_rest_bad_request(transport: str='rest', request_type=vmmigration.DeleteDatacenterConnectorRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_datacenter_connector(request)

def test_delete_datacenter_connector_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_datacenter_connector(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/datacenterConnectors/*}' % client.transport._host, args[1])

def test_delete_datacenter_connector_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_datacenter_connector(vmmigration.DeleteDatacenterConnectorRequest(), name='name_value')

def test_delete_datacenter_connector_rest_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.UpgradeApplianceRequest, dict])
def test_upgrade_appliance_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'datacenter_connector': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.upgrade_appliance(request)
    assert response.operation.name == 'operations/spam'

def test_upgrade_appliance_rest_required_fields(request_type=vmmigration.UpgradeApplianceRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['datacenter_connector'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).upgrade_appliance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['datacenterConnector'] = 'datacenter_connector_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).upgrade_appliance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'datacenterConnector' in jsonified_request
    assert jsonified_request['datacenterConnector'] == 'datacenter_connector_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.upgrade_appliance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_upgrade_appliance_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.upgrade_appliance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('datacenterConnector',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_upgrade_appliance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_upgrade_appliance') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_upgrade_appliance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.UpgradeApplianceRequest.pb(vmmigration.UpgradeApplianceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.UpgradeApplianceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.upgrade_appliance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_upgrade_appliance_rest_bad_request(transport: str='rest', request_type=vmmigration.UpgradeApplianceRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'datacenter_connector': 'projects/sample1/locations/sample2/sources/sample3/datacenterConnectors/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.upgrade_appliance(request)

def test_upgrade_appliance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateMigratingVmRequest, dict])
def test_create_migrating_vm_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request_init['migrating_vm'] = {'compute_engine_target_defaults': {'vm_name': 'vm_name_value', 'target_project': 'target_project_value', 'zone': 'zone_value', 'machine_type_series': 'machine_type_series_value', 'machine_type': 'machine_type_value', 'network_tags': ['network_tags_value1', 'network_tags_value2'], 'network_interfaces': [{'network': 'network_value', 'subnetwork': 'subnetwork_value', 'internal_ip': 'internal_ip_value', 'external_ip': 'external_ip_value'}], 'service_account': 'service_account_value', 'disk_type': 1, 'labels': {}, 'license_type': 1, 'applied_license': {'type_': 1, 'os_license': 'os_license_value'}, 'compute_scheduling': {'on_host_maintenance': 1, 'restart_type': 1, 'node_affinities': [{'key': 'key_value', 'operator': 1, 'values': ['values_value1', 'values_value2']}], 'min_node_cpus': 1379}, 'secure_boot': True, 'boot_option': 1, 'metadata': {}, 'additional_licenses': ['additional_licenses_value1', 'additional_licenses_value2'], 'hostname': 'hostname_value'}, 'aws_source_vm_details': {'firmware': 1, 'committed_storage_bytes': 2464}, 'name': 'name_value', 'source_vm_id': 'source_vm_id_value', 'display_name': 'display_name_value', 'description': 'description_value', 'policy': {'idle_duration': {'seconds': 751, 'nanos': 543}, 'skip_os_adaptation': True}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'last_sync': {'last_sync_time': {}}, 'state': 1, 'state_time': {}, 'current_sync_info': {'name': 'name_value', 'cycle_number': 1272, 'start_time': {}, 'end_time': {}, 'total_pause_duration': {}, 'progress_percent': 1733, 'steps': [{'initializing_replication': {}, 'replicating': {'total_bytes': 1194, 'replicated_bytes': 1699, 'last_two_minutes_average_bytes_per_second': 4370, 'last_thirty_minutes_average_bytes_per_second': 4700}, 'post_processing': {}, 'start_time': {}, 'end_time': {}}], 'state': 1, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}}, 'group': 'group_value', 'labels': {}, 'recent_clone_jobs': [{'compute_engine_target_details': {'vm_name': 'vm_name_value', 'project': 'project_value', 'zone': 'zone_value', 'machine_type_series': 'machine_type_series_value', 'machine_type': 'machine_type_value', 'network_tags': ['network_tags_value1', 'network_tags_value2'], 'network_interfaces': {}, 'service_account': 'service_account_value', 'disk_type': 1, 'labels': {}, 'license_type': 1, 'applied_license': {}, 'compute_scheduling': {}, 'secure_boot': True, 'boot_option': 1, 'metadata': {}, 'additional_licenses': ['additional_licenses_value1', 'additional_licenses_value2'], 'hostname': 'hostname_value'}, 'create_time': {}, 'end_time': {}, 'name': 'name_value', 'state': 1, 'state_time': {}, 'error': {}, 'steps': [{'adapting_os': {}, 'preparing_vm_disks': {}, 'instantiating_migrated_vm': {}, 'start_time': {}, 'end_time': {}}]}], 'error': {}, 'recent_cutover_jobs': [{'compute_engine_target_details': {}, 'create_time': {}, 'end_time': {}, 'name': 'name_value', 'state': 1, 'state_time': {}, 'progress_percent': 1733, 'error': {}, 'state_message': 'state_message_value', 'steps': [{'previous_replication_cycle': {}, 'shutting_down_source_vm': {}, 'final_sync': {}, 'preparing_vm_disks': {}, 'instantiating_migrated_vm': {}, 'start_time': {}, 'end_time': {}}]}]}
    test_field = vmmigration.CreateMigratingVmRequest.meta.fields['migrating_vm']

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
    for (field, value) in request_init['migrating_vm'].items():
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
                for i in range(0, len(request_init['migrating_vm'][field])):
                    del request_init['migrating_vm'][field][i][subfield]
            else:
                del request_init['migrating_vm'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_migrating_vm(request)
    assert response.operation.name == 'operations/spam'

def test_create_migrating_vm_rest_required_fields(request_type=vmmigration.CreateMigratingVmRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['migrating_vm_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'migratingVmId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_migrating_vm._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'migratingVmId' in jsonified_request
    assert jsonified_request['migratingVmId'] == request_init['migrating_vm_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['migratingVmId'] = 'migrating_vm_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_migrating_vm._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('migrating_vm_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'migratingVmId' in jsonified_request
    assert jsonified_request['migratingVmId'] == 'migrating_vm_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_migrating_vm(request)
            expected_params = [('migratingVmId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_migrating_vm_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_migrating_vm._get_unset_required_fields({})
    assert set(unset_fields) == set(('migratingVmId', 'requestId')) & set(('parent', 'migratingVmId', 'migratingVm'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_migrating_vm_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_migrating_vm') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_migrating_vm') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateMigratingVmRequest.pb(vmmigration.CreateMigratingVmRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateMigratingVmRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_migrating_vm(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_migrating_vm_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateMigratingVmRequest):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_migrating_vm(request)

def test_create_migrating_vm_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(parent='parent_value', migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), migrating_vm_id='migrating_vm_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_migrating_vm(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*}/migratingVms' % client.transport._host, args[1])

def test_create_migrating_vm_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_migrating_vm(vmmigration.CreateMigratingVmRequest(), parent='parent_value', migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), migrating_vm_id='migrating_vm_id_value')

def test_create_migrating_vm_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListMigratingVmsRequest, dict])
def test_list_migrating_vms_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListMigratingVmsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListMigratingVmsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_migrating_vms(request)
    assert isinstance(response, pagers.ListMigratingVmsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_migrating_vms_rest_required_fields(request_type=vmmigration.ListMigratingVmsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_migrating_vms._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_migrating_vms._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListMigratingVmsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListMigratingVmsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_migrating_vms(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_migrating_vms_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_migrating_vms._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken', 'view')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_migrating_vms_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_migrating_vms') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_migrating_vms') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListMigratingVmsRequest.pb(vmmigration.ListMigratingVmsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListMigratingVmsResponse.to_json(vmmigration.ListMigratingVmsResponse())
        request = vmmigration.ListMigratingVmsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListMigratingVmsResponse()
        client.list_migrating_vms(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_migrating_vms_rest_bad_request(transport: str='rest', request_type=vmmigration.ListMigratingVmsRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_migrating_vms(request)

def test_list_migrating_vms_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListMigratingVmsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListMigratingVmsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_migrating_vms(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*}/migratingVms' % client.transport._host, args[1])

def test_list_migrating_vms_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_migrating_vms(vmmigration.ListMigratingVmsRequest(), parent='parent_value')

def test_list_migrating_vms_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm(), vmmigration.MigratingVm()], next_page_token='abc'), vmmigration.ListMigratingVmsResponse(migrating_vms=[], next_page_token='def'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm()], next_page_token='ghi'), vmmigration.ListMigratingVmsResponse(migrating_vms=[vmmigration.MigratingVm(), vmmigration.MigratingVm()]))
        response = response + response
        response = tuple((vmmigration.ListMigratingVmsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3'}
        pager = client.list_migrating_vms(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.MigratingVm) for i in results))
        pages = list(client.list_migrating_vms(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetMigratingVmRequest, dict])
def test_get_migrating_vm_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.MigratingVm(name='name_value', source_vm_id='source_vm_id_value', display_name='display_name_value', description='description_value', state=vmmigration.MigratingVm.State.PENDING, group='group_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.MigratingVm.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_migrating_vm(request)
    assert isinstance(response, vmmigration.MigratingVm)
    assert response.name == 'name_value'
    assert response.source_vm_id == 'source_vm_id_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == vmmigration.MigratingVm.State.PENDING
    assert response.group == 'group_value'

def test_get_migrating_vm_rest_required_fields(request_type=vmmigration.GetMigratingVmRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_migrating_vm._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_migrating_vm._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.MigratingVm()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.MigratingVm.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_migrating_vm(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_migrating_vm_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_migrating_vm._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_migrating_vm_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_migrating_vm') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_migrating_vm') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetMigratingVmRequest.pb(vmmigration.GetMigratingVmRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.MigratingVm.to_json(vmmigration.MigratingVm())
        request = vmmigration.GetMigratingVmRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.MigratingVm()
        client.get_migrating_vm(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_migrating_vm_rest_bad_request(transport: str='rest', request_type=vmmigration.GetMigratingVmRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_migrating_vm(request)

def test_get_migrating_vm_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.MigratingVm()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.MigratingVm.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_migrating_vm(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*}' % client.transport._host, args[1])

def test_get_migrating_vm_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_migrating_vm(vmmigration.GetMigratingVmRequest(), name='name_value')

def test_get_migrating_vm_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateMigratingVmRequest, dict])
def test_update_migrating_vm_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'migrating_vm': {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}}
    request_init['migrating_vm'] = {'compute_engine_target_defaults': {'vm_name': 'vm_name_value', 'target_project': 'target_project_value', 'zone': 'zone_value', 'machine_type_series': 'machine_type_series_value', 'machine_type': 'machine_type_value', 'network_tags': ['network_tags_value1', 'network_tags_value2'], 'network_interfaces': [{'network': 'network_value', 'subnetwork': 'subnetwork_value', 'internal_ip': 'internal_ip_value', 'external_ip': 'external_ip_value'}], 'service_account': 'service_account_value', 'disk_type': 1, 'labels': {}, 'license_type': 1, 'applied_license': {'type_': 1, 'os_license': 'os_license_value'}, 'compute_scheduling': {'on_host_maintenance': 1, 'restart_type': 1, 'node_affinities': [{'key': 'key_value', 'operator': 1, 'values': ['values_value1', 'values_value2']}], 'min_node_cpus': 1379}, 'secure_boot': True, 'boot_option': 1, 'metadata': {}, 'additional_licenses': ['additional_licenses_value1', 'additional_licenses_value2'], 'hostname': 'hostname_value'}, 'aws_source_vm_details': {'firmware': 1, 'committed_storage_bytes': 2464}, 'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4', 'source_vm_id': 'source_vm_id_value', 'display_name': 'display_name_value', 'description': 'description_value', 'policy': {'idle_duration': {'seconds': 751, 'nanos': 543}, 'skip_os_adaptation': True}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'last_sync': {'last_sync_time': {}}, 'state': 1, 'state_time': {}, 'current_sync_info': {'name': 'name_value', 'cycle_number': 1272, 'start_time': {}, 'end_time': {}, 'total_pause_duration': {}, 'progress_percent': 1733, 'steps': [{'initializing_replication': {}, 'replicating': {'total_bytes': 1194, 'replicated_bytes': 1699, 'last_two_minutes_average_bytes_per_second': 4370, 'last_thirty_minutes_average_bytes_per_second': 4700}, 'post_processing': {}, 'start_time': {}, 'end_time': {}}], 'state': 1, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}}, 'group': 'group_value', 'labels': {}, 'recent_clone_jobs': [{'compute_engine_target_details': {'vm_name': 'vm_name_value', 'project': 'project_value', 'zone': 'zone_value', 'machine_type_series': 'machine_type_series_value', 'machine_type': 'machine_type_value', 'network_tags': ['network_tags_value1', 'network_tags_value2'], 'network_interfaces': {}, 'service_account': 'service_account_value', 'disk_type': 1, 'labels': {}, 'license_type': 1, 'applied_license': {}, 'compute_scheduling': {}, 'secure_boot': True, 'boot_option': 1, 'metadata': {}, 'additional_licenses': ['additional_licenses_value1', 'additional_licenses_value2'], 'hostname': 'hostname_value'}, 'create_time': {}, 'end_time': {}, 'name': 'name_value', 'state': 1, 'state_time': {}, 'error': {}, 'steps': [{'adapting_os': {}, 'preparing_vm_disks': {}, 'instantiating_migrated_vm': {}, 'start_time': {}, 'end_time': {}}]}], 'error': {}, 'recent_cutover_jobs': [{'compute_engine_target_details': {}, 'create_time': {}, 'end_time': {}, 'name': 'name_value', 'state': 1, 'state_time': {}, 'progress_percent': 1733, 'error': {}, 'state_message': 'state_message_value', 'steps': [{'previous_replication_cycle': {}, 'shutting_down_source_vm': {}, 'final_sync': {}, 'preparing_vm_disks': {}, 'instantiating_migrated_vm': {}, 'start_time': {}, 'end_time': {}}]}]}
    test_field = vmmigration.UpdateMigratingVmRequest.meta.fields['migrating_vm']

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
    for (field, value) in request_init['migrating_vm'].items():
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
                for i in range(0, len(request_init['migrating_vm'][field])):
                    del request_init['migrating_vm'][field][i][subfield]
            else:
                del request_init['migrating_vm'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_migrating_vm(request)
    assert response.operation.name == 'operations/spam'

def test_update_migrating_vm_rest_required_fields(request_type=vmmigration.UpdateMigratingVmRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_migrating_vm._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_migrating_vm._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_migrating_vm(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_migrating_vm_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_migrating_vm._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('migratingVm',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_migrating_vm_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_update_migrating_vm') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_update_migrating_vm') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.UpdateMigratingVmRequest.pb(vmmigration.UpdateMigratingVmRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.UpdateMigratingVmRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_migrating_vm(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_migrating_vm_rest_bad_request(transport: str='rest', request_type=vmmigration.UpdateMigratingVmRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'migrating_vm': {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_migrating_vm(request)

def test_update_migrating_vm_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'migrating_vm': {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}}
        mock_args = dict(migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_migrating_vm(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{migrating_vm.name=projects/*/locations/*/sources/*/migratingVms/*}' % client.transport._host, args[1])

def test_update_migrating_vm_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_migrating_vm(vmmigration.UpdateMigratingVmRequest(), migrating_vm=vmmigration.MigratingVm(compute_engine_target_defaults=vmmigration.ComputeEngineTargetDefaults(vm_name='vm_name_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_migrating_vm_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteMigratingVmRequest, dict])
def test_delete_migrating_vm_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_migrating_vm(request)
    assert response.operation.name == 'operations/spam'

def test_delete_migrating_vm_rest_required_fields(request_type=vmmigration.DeleteMigratingVmRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_migrating_vm._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_migrating_vm._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_migrating_vm(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_migrating_vm_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_migrating_vm._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_migrating_vm_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_delete_migrating_vm') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_delete_migrating_vm') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.DeleteMigratingVmRequest.pb(vmmigration.DeleteMigratingVmRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.DeleteMigratingVmRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_migrating_vm(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_migrating_vm_rest_bad_request(transport: str='rest', request_type=vmmigration.DeleteMigratingVmRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_migrating_vm(request)

def test_delete_migrating_vm_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_migrating_vm(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*}' % client.transport._host, args[1])

def test_delete_migrating_vm_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_migrating_vm(vmmigration.DeleteMigratingVmRequest(), name='name_value')

def test_delete_migrating_vm_rest_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.StartMigrationRequest, dict])
def test_start_migration_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_migration(request)
    assert response.operation.name == 'operations/spam'

def test_start_migration_rest_required_fields(request_type=vmmigration.StartMigrationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['migrating_vm'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['migratingVm'] = 'migrating_vm_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'migratingVm' in jsonified_request
    assert jsonified_request['migratingVm'] == 'migrating_vm_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.start_migration(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_start_migration_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.start_migration._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('migratingVm',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_migration_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_start_migration') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_start_migration') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.StartMigrationRequest.pb(vmmigration.StartMigrationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.StartMigrationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.start_migration(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_migration_rest_bad_request(transport: str='rest', request_type=vmmigration.StartMigrationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_migration(request)

def test_start_migration_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(migrating_vm='migrating_vm_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.start_migration(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{migrating_vm=projects/*/locations/*/sources/*/migratingVms/*}:startMigration' % client.transport._host, args[1])

def test_start_migration_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.start_migration(vmmigration.StartMigrationRequest(), migrating_vm='migrating_vm_value')

def test_start_migration_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ResumeMigrationRequest, dict])
def test_resume_migration_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_migration(request)
    assert response.operation.name == 'operations/spam'

def test_resume_migration_rest_required_fields(request_type=vmmigration.ResumeMigrationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['migrating_vm'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['migratingVm'] = 'migrating_vm_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'migratingVm' in jsonified_request
    assert jsonified_request['migratingVm'] == 'migrating_vm_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resume_migration(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_migration_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_migration._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('migratingVm',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_migration_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_resume_migration') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_resume_migration') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ResumeMigrationRequest.pb(vmmigration.ResumeMigrationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.ResumeMigrationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.resume_migration(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resume_migration_rest_bad_request(transport: str='rest', request_type=vmmigration.ResumeMigrationRequest):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_migration(request)

def test_resume_migration_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.PauseMigrationRequest, dict])
def test_pause_migration_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.pause_migration(request)
    assert response.operation.name == 'operations/spam'

def test_pause_migration_rest_required_fields(request_type=vmmigration.PauseMigrationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['migrating_vm'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['migratingVm'] = 'migrating_vm_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'migratingVm' in jsonified_request
    assert jsonified_request['migratingVm'] == 'migrating_vm_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.pause_migration(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_pause_migration_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.pause_migration._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('migratingVm',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_pause_migration_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_pause_migration') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_pause_migration') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.PauseMigrationRequest.pb(vmmigration.PauseMigrationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.PauseMigrationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.pause_migration(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_pause_migration_rest_bad_request(transport: str='rest', request_type=vmmigration.PauseMigrationRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.pause_migration(request)

def test_pause_migration_rest_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.FinalizeMigrationRequest, dict])
def test_finalize_migration_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.finalize_migration(request)
    assert response.operation.name == 'operations/spam'

def test_finalize_migration_rest_required_fields(request_type=vmmigration.FinalizeMigrationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['migrating_vm'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).finalize_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['migratingVm'] = 'migrating_vm_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).finalize_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'migratingVm' in jsonified_request
    assert jsonified_request['migratingVm'] == 'migrating_vm_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.finalize_migration(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_finalize_migration_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.finalize_migration._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('migratingVm',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_finalize_migration_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_finalize_migration') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_finalize_migration') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.FinalizeMigrationRequest.pb(vmmigration.FinalizeMigrationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.FinalizeMigrationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.finalize_migration(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_finalize_migration_rest_bad_request(transport: str='rest', request_type=vmmigration.FinalizeMigrationRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.finalize_migration(request)

def test_finalize_migration_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'migrating_vm': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(migrating_vm='migrating_vm_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.finalize_migration(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{migrating_vm=projects/*/locations/*/sources/*/migratingVms/*}:finalizeMigration' % client.transport._host, args[1])

def test_finalize_migration_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.finalize_migration(vmmigration.FinalizeMigrationRequest(), migrating_vm='migrating_vm_value')

def test_finalize_migration_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateCloneJobRequest, dict])
def test_create_clone_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request_init['clone_job'] = {'compute_engine_target_details': {'vm_name': 'vm_name_value', 'project': 'project_value', 'zone': 'zone_value', 'machine_type_series': 'machine_type_series_value', 'machine_type': 'machine_type_value', 'network_tags': ['network_tags_value1', 'network_tags_value2'], 'network_interfaces': [{'network': 'network_value', 'subnetwork': 'subnetwork_value', 'internal_ip': 'internal_ip_value', 'external_ip': 'external_ip_value'}], 'service_account': 'service_account_value', 'disk_type': 1, 'labels': {}, 'license_type': 1, 'applied_license': {'type_': 1, 'os_license': 'os_license_value'}, 'compute_scheduling': {'on_host_maintenance': 1, 'restart_type': 1, 'node_affinities': [{'key': 'key_value', 'operator': 1, 'values': ['values_value1', 'values_value2']}], 'min_node_cpus': 1379}, 'secure_boot': True, 'boot_option': 1, 'metadata': {}, 'additional_licenses': ['additional_licenses_value1', 'additional_licenses_value2'], 'hostname': 'hostname_value'}, 'create_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}, 'name': 'name_value', 'state': 1, 'state_time': {}, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'steps': [{'adapting_os': {}, 'preparing_vm_disks': {}, 'instantiating_migrated_vm': {}, 'start_time': {}, 'end_time': {}}]}
    test_field = vmmigration.CreateCloneJobRequest.meta.fields['clone_job']

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
    for (field, value) in request_init['clone_job'].items():
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
                for i in range(0, len(request_init['clone_job'][field])):
                    del request_init['clone_job'][field][i][subfield]
            else:
                del request_init['clone_job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_clone_job(request)
    assert response.operation.name == 'operations/spam'

def test_create_clone_job_rest_required_fields(request_type=vmmigration.CreateCloneJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['clone_job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'cloneJobId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_clone_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'cloneJobId' in jsonified_request
    assert jsonified_request['cloneJobId'] == request_init['clone_job_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['cloneJobId'] = 'clone_job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_clone_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('clone_job_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'cloneJobId' in jsonified_request
    assert jsonified_request['cloneJobId'] == 'clone_job_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_clone_job(request)
            expected_params = [('cloneJobId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_clone_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_clone_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('cloneJobId', 'requestId')) & set(('parent', 'cloneJobId', 'cloneJob'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_clone_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_clone_job') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_clone_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateCloneJobRequest.pb(vmmigration.CreateCloneJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateCloneJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_clone_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_clone_job_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateCloneJobRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_clone_job(request)

def test_create_clone_job_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(parent='parent_value', clone_job=vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), clone_job_id='clone_job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_clone_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*/migratingVms/*}/cloneJobs' % client.transport._host, args[1])

def test_create_clone_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_clone_job(vmmigration.CreateCloneJobRequest(), parent='parent_value', clone_job=vmmigration.CloneJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), clone_job_id='clone_job_id_value')

def test_create_clone_job_rest_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CancelCloneJobRequest, dict])
def test_cancel_clone_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cloneJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_clone_job(request)
    assert response.operation.name == 'operations/spam'

def test_cancel_clone_job_rest_required_fields(request_type=vmmigration.CancelCloneJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_clone_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_clone_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.cancel_clone_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_clone_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_clone_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_clone_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_cancel_clone_job') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_cancel_clone_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CancelCloneJobRequest.pb(vmmigration.CancelCloneJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CancelCloneJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.cancel_clone_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_cancel_clone_job_rest_bad_request(transport: str='rest', request_type=vmmigration.CancelCloneJobRequest):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cloneJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_clone_job(request)

def test_cancel_clone_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cloneJobs/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.cancel_clone_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*/cloneJobs/*}:cancel' % client.transport._host, args[1])

def test_cancel_clone_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.cancel_clone_job(vmmigration.CancelCloneJobRequest(), name='name_value')

def test_cancel_clone_job_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListCloneJobsRequest, dict])
def test_list_clone_jobs_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListCloneJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListCloneJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_clone_jobs(request)
    assert isinstance(response, pagers.ListCloneJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_clone_jobs_rest_required_fields(request_type=vmmigration.ListCloneJobsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_clone_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_clone_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListCloneJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListCloneJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_clone_jobs(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_clone_jobs_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_clone_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_clone_jobs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_clone_jobs') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_clone_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListCloneJobsRequest.pb(vmmigration.ListCloneJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListCloneJobsResponse.to_json(vmmigration.ListCloneJobsResponse())
        request = vmmigration.ListCloneJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListCloneJobsResponse()
        client.list_clone_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_clone_jobs_rest_bad_request(transport: str='rest', request_type=vmmigration.ListCloneJobsRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_clone_jobs(request)

def test_list_clone_jobs_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListCloneJobsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListCloneJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_clone_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*/migratingVms/*}/cloneJobs' % client.transport._host, args[1])

def test_list_clone_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_clone_jobs(vmmigration.ListCloneJobsRequest(), parent='parent_value')

def test_list_clone_jobs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob(), vmmigration.CloneJob()], next_page_token='abc'), vmmigration.ListCloneJobsResponse(clone_jobs=[], next_page_token='def'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob()], next_page_token='ghi'), vmmigration.ListCloneJobsResponse(clone_jobs=[vmmigration.CloneJob(), vmmigration.CloneJob()]))
        response = response + response
        response = tuple((vmmigration.ListCloneJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        pager = client.list_clone_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.CloneJob) for i in results))
        pages = list(client.list_clone_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetCloneJobRequest, dict])
def test_get_clone_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cloneJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.CloneJob(name='name_value', state=vmmigration.CloneJob.State.PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.CloneJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_clone_job(request)
    assert isinstance(response, vmmigration.CloneJob)
    assert response.name == 'name_value'
    assert response.state == vmmigration.CloneJob.State.PENDING

def test_get_clone_job_rest_required_fields(request_type=vmmigration.GetCloneJobRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_clone_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_clone_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.CloneJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.CloneJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_clone_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_clone_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_clone_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_clone_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_clone_job') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_clone_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetCloneJobRequest.pb(vmmigration.GetCloneJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.CloneJob.to_json(vmmigration.CloneJob())
        request = vmmigration.GetCloneJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.CloneJob()
        client.get_clone_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_clone_job_rest_bad_request(transport: str='rest', request_type=vmmigration.GetCloneJobRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cloneJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_clone_job(request)

def test_get_clone_job_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.CloneJob()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cloneJobs/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.CloneJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_clone_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*/cloneJobs/*}' % client.transport._host, args[1])

def test_get_clone_job_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_clone_job(vmmigration.GetCloneJobRequest(), name='name_value')

def test_get_clone_job_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateCutoverJobRequest, dict])
def test_create_cutover_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request_init['cutover_job'] = {'compute_engine_target_details': {'vm_name': 'vm_name_value', 'project': 'project_value', 'zone': 'zone_value', 'machine_type_series': 'machine_type_series_value', 'machine_type': 'machine_type_value', 'network_tags': ['network_tags_value1', 'network_tags_value2'], 'network_interfaces': [{'network': 'network_value', 'subnetwork': 'subnetwork_value', 'internal_ip': 'internal_ip_value', 'external_ip': 'external_ip_value'}], 'service_account': 'service_account_value', 'disk_type': 1, 'labels': {}, 'license_type': 1, 'applied_license': {'type_': 1, 'os_license': 'os_license_value'}, 'compute_scheduling': {'on_host_maintenance': 1, 'restart_type': 1, 'node_affinities': [{'key': 'key_value', 'operator': 1, 'values': ['values_value1', 'values_value2']}], 'min_node_cpus': 1379}, 'secure_boot': True, 'boot_option': 1, 'metadata': {}, 'additional_licenses': ['additional_licenses_value1', 'additional_licenses_value2'], 'hostname': 'hostname_value'}, 'create_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}, 'name': 'name_value', 'state': 1, 'state_time': {}, 'progress_percent': 1733, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'state_message': 'state_message_value', 'steps': [{'previous_replication_cycle': {'name': 'name_value', 'cycle_number': 1272, 'start_time': {}, 'end_time': {}, 'total_pause_duration': {'seconds': 751, 'nanos': 543}, 'progress_percent': 1733, 'steps': [{'initializing_replication': {}, 'replicating': {'total_bytes': 1194, 'replicated_bytes': 1699, 'last_two_minutes_average_bytes_per_second': 4370, 'last_thirty_minutes_average_bytes_per_second': 4700}, 'post_processing': {}, 'start_time': {}, 'end_time': {}}], 'state': 1, 'error': {}}, 'shutting_down_source_vm': {}, 'final_sync': {}, 'preparing_vm_disks': {}, 'instantiating_migrated_vm': {}, 'start_time': {}, 'end_time': {}}]}
    test_field = vmmigration.CreateCutoverJobRequest.meta.fields['cutover_job']

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
    for (field, value) in request_init['cutover_job'].items():
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
                for i in range(0, len(request_init['cutover_job'][field])):
                    del request_init['cutover_job'][field][i][subfield]
            else:
                del request_init['cutover_job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_cutover_job(request)
    assert response.operation.name == 'operations/spam'

def test_create_cutover_job_rest_required_fields(request_type=vmmigration.CreateCutoverJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['cutover_job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'cutoverJobId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_cutover_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'cutoverJobId' in jsonified_request
    assert jsonified_request['cutoverJobId'] == request_init['cutover_job_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['cutoverJobId'] = 'cutover_job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_cutover_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('cutover_job_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'cutoverJobId' in jsonified_request
    assert jsonified_request['cutoverJobId'] == 'cutover_job_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_cutover_job(request)
            expected_params = [('cutoverJobId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_cutover_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_cutover_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('cutoverJobId', 'requestId')) & set(('parent', 'cutoverJobId', 'cutoverJob'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_cutover_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_cutover_job') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_cutover_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateCutoverJobRequest.pb(vmmigration.CreateCutoverJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateCutoverJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_cutover_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_cutover_job_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateCutoverJobRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_cutover_job(request)

def test_create_cutover_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(parent='parent_value', cutover_job=vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), cutover_job_id='cutover_job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_cutover_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*/migratingVms/*}/cutoverJobs' % client.transport._host, args[1])

def test_create_cutover_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_cutover_job(vmmigration.CreateCutoverJobRequest(), parent='parent_value', cutover_job=vmmigration.CutoverJob(compute_engine_target_details=vmmigration.ComputeEngineTargetDetails(vm_name='vm_name_value')), cutover_job_id='cutover_job_id_value')

def test_create_cutover_job_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CancelCutoverJobRequest, dict])
def test_cancel_cutover_job_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cutoverJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_cutover_job(request)
    assert response.operation.name == 'operations/spam'

def test_cancel_cutover_job_rest_required_fields(request_type=vmmigration.CancelCutoverJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_cutover_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_cutover_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.cancel_cutover_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_cutover_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_cutover_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_cutover_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_cancel_cutover_job') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_cancel_cutover_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CancelCutoverJobRequest.pb(vmmigration.CancelCutoverJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CancelCutoverJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.cancel_cutover_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_cancel_cutover_job_rest_bad_request(transport: str='rest', request_type=vmmigration.CancelCutoverJobRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cutoverJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_cutover_job(request)

def test_cancel_cutover_job_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cutoverJobs/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.cancel_cutover_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*/cutoverJobs/*}:cancel' % client.transport._host, args[1])

def test_cancel_cutover_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.cancel_cutover_job(vmmigration.CancelCutoverJobRequest(), name='name_value')

def test_cancel_cutover_job_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListCutoverJobsRequest, dict])
def test_list_cutover_jobs_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListCutoverJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListCutoverJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_cutover_jobs(request)
    assert isinstance(response, pagers.ListCutoverJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_cutover_jobs_rest_required_fields(request_type=vmmigration.ListCutoverJobsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_cutover_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_cutover_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListCutoverJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListCutoverJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_cutover_jobs(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_cutover_jobs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_cutover_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_cutover_jobs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_cutover_jobs') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_cutover_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListCutoverJobsRequest.pb(vmmigration.ListCutoverJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListCutoverJobsResponse.to_json(vmmigration.ListCutoverJobsResponse())
        request = vmmigration.ListCutoverJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListCutoverJobsResponse()
        client.list_cutover_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_cutover_jobs_rest_bad_request(transport: str='rest', request_type=vmmigration.ListCutoverJobsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_cutover_jobs(request)

def test_list_cutover_jobs_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListCutoverJobsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListCutoverJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_cutover_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*/migratingVms/*}/cutoverJobs' % client.transport._host, args[1])

def test_list_cutover_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_cutover_jobs(vmmigration.ListCutoverJobsRequest(), parent='parent_value')

def test_list_cutover_jobs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob(), vmmigration.CutoverJob()], next_page_token='abc'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[], next_page_token='def'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob()], next_page_token='ghi'), vmmigration.ListCutoverJobsResponse(cutover_jobs=[vmmigration.CutoverJob(), vmmigration.CutoverJob()]))
        response = response + response
        response = tuple((vmmigration.ListCutoverJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        pager = client.list_cutover_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.CutoverJob) for i in results))
        pages = list(client.list_cutover_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetCutoverJobRequest, dict])
def test_get_cutover_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cutoverJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.CutoverJob(name='name_value', state=vmmigration.CutoverJob.State.PENDING, progress_percent=1733, state_message='state_message_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.CutoverJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_cutover_job(request)
    assert isinstance(response, vmmigration.CutoverJob)
    assert response.name == 'name_value'
    assert response.state == vmmigration.CutoverJob.State.PENDING
    assert response.progress_percent == 1733
    assert response.state_message == 'state_message_value'

def test_get_cutover_job_rest_required_fields(request_type=vmmigration.GetCutoverJobRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_cutover_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_cutover_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.CutoverJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.CutoverJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_cutover_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_cutover_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_cutover_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_cutover_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_cutover_job') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_cutover_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetCutoverJobRequest.pb(vmmigration.GetCutoverJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.CutoverJob.to_json(vmmigration.CutoverJob())
        request = vmmigration.GetCutoverJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.CutoverJob()
        client.get_cutover_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_cutover_job_rest_bad_request(transport: str='rest', request_type=vmmigration.GetCutoverJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cutoverJobs/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_cutover_job(request)

def test_get_cutover_job_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.CutoverJob()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/cutoverJobs/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.CutoverJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_cutover_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*/cutoverJobs/*}' % client.transport._host, args[1])

def test_get_cutover_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_cutover_job(vmmigration.GetCutoverJobRequest(), name='name_value')

def test_get_cutover_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListGroupsRequest, dict])
def test_list_groups_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListGroupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListGroupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_groups(request)
    assert isinstance(response, pagers.ListGroupsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_groups_rest_required_fields(request_type=vmmigration.ListGroupsRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_groups._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_groups._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListGroupsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListGroupsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_groups(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_groups_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_groups._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_groups_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_groups') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_groups') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListGroupsRequest.pb(vmmigration.ListGroupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListGroupsResponse.to_json(vmmigration.ListGroupsResponse())
        request = vmmigration.ListGroupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListGroupsResponse()
        client.list_groups(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_groups_rest_bad_request(transport: str='rest', request_type=vmmigration.ListGroupsRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_groups(request)

def test_list_groups_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListGroupsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListGroupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_groups(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/groups' % client.transport._host, args[1])

def test_list_groups_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_groups(vmmigration.ListGroupsRequest(), parent='parent_value')

def test_list_groups_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group(), vmmigration.Group()], next_page_token='abc'), vmmigration.ListGroupsResponse(groups=[], next_page_token='def'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group()], next_page_token='ghi'), vmmigration.ListGroupsResponse(groups=[vmmigration.Group(), vmmigration.Group()]))
        response = response + response
        response = tuple((vmmigration.ListGroupsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_groups(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.Group) for i in results))
        pages = list(client.list_groups(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetGroupRequest, dict])
def test_get_group_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.Group(name='name_value', description='description_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.Group.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_group(request)
    assert isinstance(response, vmmigration.Group)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'

def test_get_group_rest_required_fields(request_type=vmmigration.GetGroupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.Group()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.Group.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_group_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_group._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_group_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_group') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetGroupRequest.pb(vmmigration.GetGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.Group.to_json(vmmigration.Group())
        request = vmmigration.GetGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.Group()
        client.get_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_group_rest_bad_request(transport: str='rest', request_type=vmmigration.GetGroupRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_group(request)

def test_get_group_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.Group()
        sample_request = {'name': 'projects/sample1/locations/sample2/groups/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.Group.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/groups/*}' % client.transport._host, args[1])

def test_get_group_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_group(vmmigration.GetGroupRequest(), name='name_value')

def test_get_group_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateGroupRequest, dict])
def test_create_group_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['group'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'display_name': 'display_name_value'}
    test_field = vmmigration.CreateGroupRequest.meta.fields['group']

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
    for (field, value) in request_init['group'].items():
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
                for i in range(0, len(request_init['group'][field])):
                    del request_init['group'][field][i][subfield]
            else:
                del request_init['group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_group(request)
    assert response.operation.name == 'operations/spam'

def test_create_group_rest_required_fields(request_type=vmmigration.CreateGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['group_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'groupId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'groupId' in jsonified_request
    assert jsonified_request['groupId'] == request_init['group_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['groupId'] = 'group_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('group_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'groupId' in jsonified_request
    assert jsonified_request['groupId'] == 'group_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_group(request)
            expected_params = [('groupId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('groupId', 'requestId')) & set(('parent', 'groupId', 'group'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_group_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_group') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateGroupRequest.pb(vmmigration.CreateGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_group_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateGroupRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_group(request)

def test_create_group_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', group=vmmigration.Group(name='name_value'), group_id='group_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/groups' % client.transport._host, args[1])

def test_create_group_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_group(vmmigration.CreateGroupRequest(), parent='parent_value', group=vmmigration.Group(name='name_value'), group_id='group_id_value')

def test_create_group_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateGroupRequest, dict])
def test_update_group_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'group': {'name': 'projects/sample1/locations/sample2/groups/sample3'}}
    request_init['group'] = {'name': 'projects/sample1/locations/sample2/groups/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'display_name': 'display_name_value'}
    test_field = vmmigration.UpdateGroupRequest.meta.fields['group']

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
    for (field, value) in request_init['group'].items():
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
                for i in range(0, len(request_init['group'][field])):
                    del request_init['group'][field][i][subfield]
            else:
                del request_init['group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_group(request)
    assert response.operation.name == 'operations/spam'

def test_update_group_rest_required_fields(request_type=vmmigration.UpdateGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_group_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('group',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_group_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_update_group') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_update_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.UpdateGroupRequest.pb(vmmigration.UpdateGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.UpdateGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_group_rest_bad_request(transport: str='rest', request_type=vmmigration.UpdateGroupRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'group': {'name': 'projects/sample1/locations/sample2/groups/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_group(request)

def test_update_group_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'group': {'name': 'projects/sample1/locations/sample2/groups/sample3'}}
        mock_args = dict(group=vmmigration.Group(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{group.name=projects/*/locations/*/groups/*}' % client.transport._host, args[1])

def test_update_group_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_group(vmmigration.UpdateGroupRequest(), group=vmmigration.Group(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_group_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteGroupRequest, dict])
def test_delete_group_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_group(request)
    assert response.operation.name == 'operations/spam'

def test_delete_group_rest_required_fields(request_type=vmmigration.DeleteGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_group_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_group_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_delete_group') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_delete_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.DeleteGroupRequest.pb(vmmigration.DeleteGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.DeleteGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_group_rest_bad_request(transport: str='rest', request_type=vmmigration.DeleteGroupRequest):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_group(request)

def test_delete_group_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/groups/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/groups/*}' % client.transport._host, args[1])

def test_delete_group_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_group(vmmigration.DeleteGroupRequest(), name='name_value')

def test_delete_group_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.AddGroupMigrationRequest, dict])
def test_add_group_migration_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'group': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_group_migration(request)
    assert response.operation.name == 'operations/spam'

def test_add_group_migration_rest_required_fields(request_type=vmmigration.AddGroupMigrationRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['group'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_group_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['group'] = 'group_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_group_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'group' in jsonified_request
    assert jsonified_request['group'] == 'group_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_group_migration(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_group_migration_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_group_migration._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('group',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_group_migration_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_add_group_migration') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_add_group_migration') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.AddGroupMigrationRequest.pb(vmmigration.AddGroupMigrationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.AddGroupMigrationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.add_group_migration(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_group_migration_rest_bad_request(transport: str='rest', request_type=vmmigration.AddGroupMigrationRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'group': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_group_migration(request)

def test_add_group_migration_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'group': 'projects/sample1/locations/sample2/groups/sample3'}
        mock_args = dict(group='group_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_group_migration(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{group=projects/*/locations/*/groups/*}:addGroupMigration' % client.transport._host, args[1])

def test_add_group_migration_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_group_migration(vmmigration.AddGroupMigrationRequest(), group='group_value')

def test_add_group_migration_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.RemoveGroupMigrationRequest, dict])
def test_remove_group_migration_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'group': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_group_migration(request)
    assert response.operation.name == 'operations/spam'

def test_remove_group_migration_rest_required_fields(request_type=vmmigration.RemoveGroupMigrationRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['group'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_group_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['group'] = 'group_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_group_migration._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'group' in jsonified_request
    assert jsonified_request['group'] == 'group_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_group_migration(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_group_migration_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_group_migration._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('group',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_group_migration_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_remove_group_migration') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_remove_group_migration') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.RemoveGroupMigrationRequest.pb(vmmigration.RemoveGroupMigrationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.RemoveGroupMigrationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.remove_group_migration(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_group_migration_rest_bad_request(transport: str='rest', request_type=vmmigration.RemoveGroupMigrationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'group': 'projects/sample1/locations/sample2/groups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_group_migration(request)

def test_remove_group_migration_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'group': 'projects/sample1/locations/sample2/groups/sample3'}
        mock_args = dict(group='group_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_group_migration(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{group=projects/*/locations/*/groups/*}:removeGroupMigration' % client.transport._host, args[1])

def test_remove_group_migration_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_group_migration(vmmigration.RemoveGroupMigrationRequest(), group='group_value')

def test_remove_group_migration_rest_error():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListTargetProjectsRequest, dict])
def test_list_target_projects_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListTargetProjectsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListTargetProjectsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_target_projects(request)
    assert isinstance(response, pagers.ListTargetProjectsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_target_projects_rest_required_fields(request_type=vmmigration.ListTargetProjectsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_target_projects._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_target_projects._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListTargetProjectsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListTargetProjectsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_target_projects(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_target_projects_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_target_projects._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_target_projects_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_target_projects') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_target_projects') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListTargetProjectsRequest.pb(vmmigration.ListTargetProjectsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListTargetProjectsResponse.to_json(vmmigration.ListTargetProjectsResponse())
        request = vmmigration.ListTargetProjectsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListTargetProjectsResponse()
        client.list_target_projects(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_target_projects_rest_bad_request(transport: str='rest', request_type=vmmigration.ListTargetProjectsRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_target_projects(request)

def test_list_target_projects_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListTargetProjectsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListTargetProjectsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_target_projects(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/targetProjects' % client.transport._host, args[1])

def test_list_target_projects_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_target_projects(vmmigration.ListTargetProjectsRequest(), parent='parent_value')

def test_list_target_projects_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject(), vmmigration.TargetProject()], next_page_token='abc'), vmmigration.ListTargetProjectsResponse(target_projects=[], next_page_token='def'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject()], next_page_token='ghi'), vmmigration.ListTargetProjectsResponse(target_projects=[vmmigration.TargetProject(), vmmigration.TargetProject()]))
        response = response + response
        response = tuple((vmmigration.ListTargetProjectsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_target_projects(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.TargetProject) for i in results))
        pages = list(client.list_target_projects(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetTargetProjectRequest, dict])
def test_get_target_project_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.TargetProject(name='name_value', project='project_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.TargetProject.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_target_project(request)
    assert isinstance(response, vmmigration.TargetProject)
    assert response.name == 'name_value'
    assert response.project == 'project_value'
    assert response.description == 'description_value'

def test_get_target_project_rest_required_fields(request_type=vmmigration.GetTargetProjectRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_target_project._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_target_project._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.TargetProject()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.TargetProject.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_target_project(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_target_project_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_target_project._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_target_project_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_target_project') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_target_project') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetTargetProjectRequest.pb(vmmigration.GetTargetProjectRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.TargetProject.to_json(vmmigration.TargetProject())
        request = vmmigration.GetTargetProjectRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.TargetProject()
        client.get_target_project(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_target_project_rest_bad_request(transport: str='rest', request_type=vmmigration.GetTargetProjectRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_target_project(request)

def test_get_target_project_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.TargetProject()
        sample_request = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.TargetProject.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_target_project(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/targetProjects/*}' % client.transport._host, args[1])

def test_get_target_project_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_target_project(vmmigration.GetTargetProjectRequest(), name='name_value')

def test_get_target_project_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.CreateTargetProjectRequest, dict])
def test_create_target_project_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['target_project'] = {'name': 'name_value', 'project': 'project_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}}
    test_field = vmmigration.CreateTargetProjectRequest.meta.fields['target_project']

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
    for (field, value) in request_init['target_project'].items():
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
                for i in range(0, len(request_init['target_project'][field])):
                    del request_init['target_project'][field][i][subfield]
            else:
                del request_init['target_project'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_target_project(request)
    assert response.operation.name == 'operations/spam'

def test_create_target_project_rest_required_fields(request_type=vmmigration.CreateTargetProjectRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['target_project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'targetProjectId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_target_project._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'targetProjectId' in jsonified_request
    assert jsonified_request['targetProjectId'] == request_init['target_project_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['targetProjectId'] = 'target_project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_target_project._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'target_project_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'targetProjectId' in jsonified_request
    assert jsonified_request['targetProjectId'] == 'target_project_id_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_target_project(request)
            expected_params = [('targetProjectId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_target_project_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_target_project._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'targetProjectId')) & set(('parent', 'targetProjectId', 'targetProject'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_target_project_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_create_target_project') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_create_target_project') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.CreateTargetProjectRequest.pb(vmmigration.CreateTargetProjectRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.CreateTargetProjectRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_target_project(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_target_project_rest_bad_request(transport: str='rest', request_type=vmmigration.CreateTargetProjectRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_target_project(request)

def test_create_target_project_rest_flattened():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', target_project=vmmigration.TargetProject(name='name_value'), target_project_id='target_project_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_target_project(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/targetProjects' % client.transport._host, args[1])

def test_create_target_project_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_target_project(vmmigration.CreateTargetProjectRequest(), parent='parent_value', target_project=vmmigration.TargetProject(name='name_value'), target_project_id='target_project_id_value')

def test_create_target_project_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.UpdateTargetProjectRequest, dict])
def test_update_target_project_rest(request_type):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'target_project': {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}}
    request_init['target_project'] = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3', 'project': 'project_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}}
    test_field = vmmigration.UpdateTargetProjectRequest.meta.fields['target_project']

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
    for (field, value) in request_init['target_project'].items():
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
                for i in range(0, len(request_init['target_project'][field])):
                    del request_init['target_project'][field][i][subfield]
            else:
                del request_init['target_project'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_target_project(request)
    assert response.operation.name == 'operations/spam'

def test_update_target_project_rest_required_fields(request_type=vmmigration.UpdateTargetProjectRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_target_project._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_target_project._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_target_project(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_target_project_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_target_project._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('targetProject',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_target_project_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_update_target_project') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_update_target_project') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.UpdateTargetProjectRequest.pb(vmmigration.UpdateTargetProjectRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.UpdateTargetProjectRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_target_project(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_target_project_rest_bad_request(transport: str='rest', request_type=vmmigration.UpdateTargetProjectRequest):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'target_project': {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_target_project(request)

def test_update_target_project_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'target_project': {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}}
        mock_args = dict(target_project=vmmigration.TargetProject(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_target_project(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{target_project.name=projects/*/locations/*/targetProjects/*}' % client.transport._host, args[1])

def test_update_target_project_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_target_project(vmmigration.UpdateTargetProjectRequest(), target_project=vmmigration.TargetProject(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_target_project_rest_error():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.DeleteTargetProjectRequest, dict])
def test_delete_target_project_rest(request_type):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_target_project(request)
    assert response.operation.name == 'operations/spam'

def test_delete_target_project_rest_required_fields(request_type=vmmigration.DeleteTargetProjectRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_target_project._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_target_project._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_target_project(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_target_project_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_target_project._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_target_project_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmMigrationRestInterceptor, 'post_delete_target_project') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_delete_target_project') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.DeleteTargetProjectRequest.pb(vmmigration.DeleteTargetProjectRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmmigration.DeleteTargetProjectRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_target_project(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_target_project_rest_bad_request(transport: str='rest', request_type=vmmigration.DeleteTargetProjectRequest):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_target_project(request)

def test_delete_target_project_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/targetProjects/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_target_project(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/targetProjects/*}' % client.transport._host, args[1])

def test_delete_target_project_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_target_project(vmmigration.DeleteTargetProjectRequest(), name='name_value')

def test_delete_target_project_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmmigration.ListReplicationCyclesRequest, dict])
def test_list_replication_cycles_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListReplicationCyclesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListReplicationCyclesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_replication_cycles(request)
    assert isinstance(response, pagers.ListReplicationCyclesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_replication_cycles_rest_required_fields(request_type=vmmigration.ListReplicationCyclesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_token'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageToken' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_replication_cycles._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == request_init['page_token']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageToken'] = 'page_token_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_replication_cycles._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageToken' in jsonified_request
    assert jsonified_request['pageToken'] == 'page_token_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ListReplicationCyclesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ListReplicationCyclesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_replication_cycles(request)
            expected_params = [('pageToken', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_replication_cycles_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_replication_cycles._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent', 'pageToken'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_replication_cycles_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_list_replication_cycles') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_list_replication_cycles') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.ListReplicationCyclesRequest.pb(vmmigration.ListReplicationCyclesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ListReplicationCyclesResponse.to_json(vmmigration.ListReplicationCyclesResponse())
        request = vmmigration.ListReplicationCyclesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ListReplicationCyclesResponse()
        client.list_replication_cycles(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_replication_cycles_rest_bad_request(transport: str='rest', request_type=vmmigration.ListReplicationCyclesRequest):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_replication_cycles(request)

def test_list_replication_cycles_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ListReplicationCyclesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ListReplicationCyclesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_replication_cycles(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/sources/*/migratingVms/*}/replicationCycles' % client.transport._host, args[1])

def test_list_replication_cycles_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_replication_cycles(vmmigration.ListReplicationCyclesRequest(), parent='parent_value')

def test_list_replication_cycles_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()], next_page_token='abc'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[], next_page_token='def'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle()], next_page_token='ghi'), vmmigration.ListReplicationCyclesResponse(replication_cycles=[vmmigration.ReplicationCycle(), vmmigration.ReplicationCycle()]))
        response = response + response
        response = tuple((vmmigration.ListReplicationCyclesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4'}
        pager = client.list_replication_cycles(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmmigration.ReplicationCycle) for i in results))
        pages = list(client.list_replication_cycles(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmmigration.GetReplicationCycleRequest, dict])
def test_get_replication_cycle_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/replicationCycles/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ReplicationCycle(name='name_value', cycle_number=1272, progress_percent=1733, state=vmmigration.ReplicationCycle.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ReplicationCycle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_replication_cycle(request)
    assert isinstance(response, vmmigration.ReplicationCycle)
    assert response.name == 'name_value'
    assert response.cycle_number == 1272
    assert response.progress_percent == 1733
    assert response.state == vmmigration.ReplicationCycle.State.RUNNING

def test_get_replication_cycle_rest_required_fields(request_type=vmmigration.GetReplicationCycleRequest):
    if False:
        return 10
    transport_class = transports.VmMigrationRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_replication_cycle._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_replication_cycle._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmmigration.ReplicationCycle()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmmigration.ReplicationCycle.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_replication_cycle(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_replication_cycle_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_replication_cycle._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_replication_cycle_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmMigrationRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmMigrationRestInterceptor())
    client = VmMigrationClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmMigrationRestInterceptor, 'post_get_replication_cycle') as post, mock.patch.object(transports.VmMigrationRestInterceptor, 'pre_get_replication_cycle') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmmigration.GetReplicationCycleRequest.pb(vmmigration.GetReplicationCycleRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmmigration.ReplicationCycle.to_json(vmmigration.ReplicationCycle())
        request = vmmigration.GetReplicationCycleRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmmigration.ReplicationCycle()
        client.get_replication_cycle(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_replication_cycle_rest_bad_request(transport: str='rest', request_type=vmmigration.GetReplicationCycleRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/replicationCycles/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_replication_cycle(request)

def test_get_replication_cycle_rest_flattened():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmmigration.ReplicationCycle()
        sample_request = {'name': 'projects/sample1/locations/sample2/sources/sample3/migratingVms/sample4/replicationCycles/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmmigration.ReplicationCycle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_replication_cycle(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/sources/*/migratingVms/*/replicationCycles/*}' % client.transport._host, args[1])

def test_get_replication_cycle_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_replication_cycle(vmmigration.GetReplicationCycleRequest(), name='name_value')

def test_get_replication_cycle_rest_error():
    if False:
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.VmMigrationGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.VmMigrationGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VmMigrationClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.VmMigrationGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VmMigrationClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VmMigrationClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.VmMigrationGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VmMigrationClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmMigrationGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = VmMigrationClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.VmMigrationGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.VmMigrationGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.VmMigrationGrpcTransport, transports.VmMigrationGrpcAsyncIOTransport, transports.VmMigrationRestTransport])
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
    transport = VmMigrationClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.VmMigrationGrpcTransport)

def test_vm_migration_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.VmMigrationTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_vm_migration_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.vmmigration_v1.services.vm_migration.transports.VmMigrationTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.VmMigrationTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_sources', 'get_source', 'create_source', 'update_source', 'delete_source', 'fetch_inventory', 'list_utilization_reports', 'get_utilization_report', 'create_utilization_report', 'delete_utilization_report', 'list_datacenter_connectors', 'get_datacenter_connector', 'create_datacenter_connector', 'delete_datacenter_connector', 'upgrade_appliance', 'create_migrating_vm', 'list_migrating_vms', 'get_migrating_vm', 'update_migrating_vm', 'delete_migrating_vm', 'start_migration', 'resume_migration', 'pause_migration', 'finalize_migration', 'create_clone_job', 'cancel_clone_job', 'list_clone_jobs', 'get_clone_job', 'create_cutover_job', 'cancel_cutover_job', 'list_cutover_jobs', 'get_cutover_job', 'list_groups', 'get_group', 'create_group', 'update_group', 'delete_group', 'add_group_migration', 'remove_group_migration', 'list_target_projects', 'get_target_project', 'create_target_project', 'update_target_project', 'delete_target_project', 'list_replication_cycles', 'get_replication_cycle', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_vm_migration_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.vmmigration_v1.services.vm_migration.transports.VmMigrationTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VmMigrationTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_vm_migration_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.vmmigration_v1.services.vm_migration.transports.VmMigrationTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VmMigrationTransport()
        adc.assert_called_once()

def test_vm_migration_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        VmMigrationClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.VmMigrationGrpcTransport, transports.VmMigrationGrpcAsyncIOTransport])
def test_vm_migration_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.VmMigrationGrpcTransport, transports.VmMigrationGrpcAsyncIOTransport, transports.VmMigrationRestTransport])
def test_vm_migration_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.VmMigrationGrpcTransport, grpc_helpers), (transports.VmMigrationGrpcAsyncIOTransport, grpc_helpers_async)])
def test_vm_migration_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('vmmigration.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='vmmigration.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.VmMigrationGrpcTransport, transports.VmMigrationGrpcAsyncIOTransport])
def test_vm_migration_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_vm_migration_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.VmMigrationRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_vm_migration_rest_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_vm_migration_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vmmigration.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('vmmigration.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmmigration.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_vm_migration_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vmmigration.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('vmmigration.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmmigration.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_vm_migration_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = VmMigrationClient(credentials=creds1, transport=transport_name)
    client2 = VmMigrationClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_sources._session
    session2 = client2.transport.list_sources._session
    assert session1 != session2
    session1 = client1.transport.get_source._session
    session2 = client2.transport.get_source._session
    assert session1 != session2
    session1 = client1.transport.create_source._session
    session2 = client2.transport.create_source._session
    assert session1 != session2
    session1 = client1.transport.update_source._session
    session2 = client2.transport.update_source._session
    assert session1 != session2
    session1 = client1.transport.delete_source._session
    session2 = client2.transport.delete_source._session
    assert session1 != session2
    session1 = client1.transport.fetch_inventory._session
    session2 = client2.transport.fetch_inventory._session
    assert session1 != session2
    session1 = client1.transport.list_utilization_reports._session
    session2 = client2.transport.list_utilization_reports._session
    assert session1 != session2
    session1 = client1.transport.get_utilization_report._session
    session2 = client2.transport.get_utilization_report._session
    assert session1 != session2
    session1 = client1.transport.create_utilization_report._session
    session2 = client2.transport.create_utilization_report._session
    assert session1 != session2
    session1 = client1.transport.delete_utilization_report._session
    session2 = client2.transport.delete_utilization_report._session
    assert session1 != session2
    session1 = client1.transport.list_datacenter_connectors._session
    session2 = client2.transport.list_datacenter_connectors._session
    assert session1 != session2
    session1 = client1.transport.get_datacenter_connector._session
    session2 = client2.transport.get_datacenter_connector._session
    assert session1 != session2
    session1 = client1.transport.create_datacenter_connector._session
    session2 = client2.transport.create_datacenter_connector._session
    assert session1 != session2
    session1 = client1.transport.delete_datacenter_connector._session
    session2 = client2.transport.delete_datacenter_connector._session
    assert session1 != session2
    session1 = client1.transport.upgrade_appliance._session
    session2 = client2.transport.upgrade_appliance._session
    assert session1 != session2
    session1 = client1.transport.create_migrating_vm._session
    session2 = client2.transport.create_migrating_vm._session
    assert session1 != session2
    session1 = client1.transport.list_migrating_vms._session
    session2 = client2.transport.list_migrating_vms._session
    assert session1 != session2
    session1 = client1.transport.get_migrating_vm._session
    session2 = client2.transport.get_migrating_vm._session
    assert session1 != session2
    session1 = client1.transport.update_migrating_vm._session
    session2 = client2.transport.update_migrating_vm._session
    assert session1 != session2
    session1 = client1.transport.delete_migrating_vm._session
    session2 = client2.transport.delete_migrating_vm._session
    assert session1 != session2
    session1 = client1.transport.start_migration._session
    session2 = client2.transport.start_migration._session
    assert session1 != session2
    session1 = client1.transport.resume_migration._session
    session2 = client2.transport.resume_migration._session
    assert session1 != session2
    session1 = client1.transport.pause_migration._session
    session2 = client2.transport.pause_migration._session
    assert session1 != session2
    session1 = client1.transport.finalize_migration._session
    session2 = client2.transport.finalize_migration._session
    assert session1 != session2
    session1 = client1.transport.create_clone_job._session
    session2 = client2.transport.create_clone_job._session
    assert session1 != session2
    session1 = client1.transport.cancel_clone_job._session
    session2 = client2.transport.cancel_clone_job._session
    assert session1 != session2
    session1 = client1.transport.list_clone_jobs._session
    session2 = client2.transport.list_clone_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_clone_job._session
    session2 = client2.transport.get_clone_job._session
    assert session1 != session2
    session1 = client1.transport.create_cutover_job._session
    session2 = client2.transport.create_cutover_job._session
    assert session1 != session2
    session1 = client1.transport.cancel_cutover_job._session
    session2 = client2.transport.cancel_cutover_job._session
    assert session1 != session2
    session1 = client1.transport.list_cutover_jobs._session
    session2 = client2.transport.list_cutover_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_cutover_job._session
    session2 = client2.transport.get_cutover_job._session
    assert session1 != session2
    session1 = client1.transport.list_groups._session
    session2 = client2.transport.list_groups._session
    assert session1 != session2
    session1 = client1.transport.get_group._session
    session2 = client2.transport.get_group._session
    assert session1 != session2
    session1 = client1.transport.create_group._session
    session2 = client2.transport.create_group._session
    assert session1 != session2
    session1 = client1.transport.update_group._session
    session2 = client2.transport.update_group._session
    assert session1 != session2
    session1 = client1.transport.delete_group._session
    session2 = client2.transport.delete_group._session
    assert session1 != session2
    session1 = client1.transport.add_group_migration._session
    session2 = client2.transport.add_group_migration._session
    assert session1 != session2
    session1 = client1.transport.remove_group_migration._session
    session2 = client2.transport.remove_group_migration._session
    assert session1 != session2
    session1 = client1.transport.list_target_projects._session
    session2 = client2.transport.list_target_projects._session
    assert session1 != session2
    session1 = client1.transport.get_target_project._session
    session2 = client2.transport.get_target_project._session
    assert session1 != session2
    session1 = client1.transport.create_target_project._session
    session2 = client2.transport.create_target_project._session
    assert session1 != session2
    session1 = client1.transport.update_target_project._session
    session2 = client2.transport.update_target_project._session
    assert session1 != session2
    session1 = client1.transport.delete_target_project._session
    session2 = client2.transport.delete_target_project._session
    assert session1 != session2
    session1 = client1.transport.list_replication_cycles._session
    session2 = client2.transport.list_replication_cycles._session
    assert session1 != session2
    session1 = client1.transport.get_replication_cycle._session
    session2 = client2.transport.get_replication_cycle._session
    assert session1 != session2

def test_vm_migration_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VmMigrationGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_vm_migration_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VmMigrationGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.VmMigrationGrpcTransport, transports.VmMigrationGrpcAsyncIOTransport])
def test_vm_migration_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.VmMigrationGrpcTransport, transports.VmMigrationGrpcAsyncIOTransport])
def test_vm_migration_transport_channel_mtls_with_adc(transport_class):
    if False:
        while True:
            i = 10
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

def test_vm_migration_grpc_lro_client():
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_vm_migration_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_clone_job_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    source = 'whelk'
    migrating_vm = 'octopus'
    clone_job = 'oyster'
    expected = 'projects/{project}/locations/{location}/sources/{source}/migratingVms/{migrating_vm}/cloneJobs/{clone_job}'.format(project=project, location=location, source=source, migrating_vm=migrating_vm, clone_job=clone_job)
    actual = VmMigrationClient.clone_job_path(project, location, source, migrating_vm, clone_job)
    assert expected == actual

def test_parse_clone_job_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'source': 'mussel', 'migrating_vm': 'winkle', 'clone_job': 'nautilus'}
    path = VmMigrationClient.clone_job_path(**expected)
    actual = VmMigrationClient.parse_clone_job_path(path)
    assert expected == actual

def test_cutover_job_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    location = 'abalone'
    source = 'squid'
    migrating_vm = 'clam'
    cutover_job = 'whelk'
    expected = 'projects/{project}/locations/{location}/sources/{source}/migratingVms/{migrating_vm}/cutoverJobs/{cutover_job}'.format(project=project, location=location, source=source, migrating_vm=migrating_vm, cutover_job=cutover_job)
    actual = VmMigrationClient.cutover_job_path(project, location, source, migrating_vm, cutover_job)
    assert expected == actual

def test_parse_cutover_job_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'source': 'nudibranch', 'migrating_vm': 'cuttlefish', 'cutover_job': 'mussel'}
    path = VmMigrationClient.cutover_job_path(**expected)
    actual = VmMigrationClient.parse_cutover_job_path(path)
    assert expected == actual

def test_datacenter_connector_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    source = 'scallop'
    datacenter_connector = 'abalone'
    expected = 'projects/{project}/locations/{location}/sources/{source}/datacenterConnectors/{datacenter_connector}'.format(project=project, location=location, source=source, datacenter_connector=datacenter_connector)
    actual = VmMigrationClient.datacenter_connector_path(project, location, source, datacenter_connector)
    assert expected == actual

def test_parse_datacenter_connector_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam', 'source': 'whelk', 'datacenter_connector': 'octopus'}
    path = VmMigrationClient.datacenter_connector_path(**expected)
    actual = VmMigrationClient.parse_datacenter_connector_path(path)
    assert expected == actual

def test_group_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    group = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/groups/{group}'.format(project=project, location=location, group=group)
    actual = VmMigrationClient.group_path(project, location, group)
    assert expected == actual

def test_parse_group_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel', 'location': 'winkle', 'group': 'nautilus'}
    path = VmMigrationClient.group_path(**expected)
    actual = VmMigrationClient.parse_group_path(path)
    assert expected == actual

def test_migrating_vm_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    source = 'squid'
    migrating_vm = 'clam'
    expected = 'projects/{project}/locations/{location}/sources/{source}/migratingVms/{migrating_vm}'.format(project=project, location=location, source=source, migrating_vm=migrating_vm)
    actual = VmMigrationClient.migrating_vm_path(project, location, source, migrating_vm)
    assert expected == actual

def test_parse_migrating_vm_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'location': 'octopus', 'source': 'oyster', 'migrating_vm': 'nudibranch'}
    path = VmMigrationClient.migrating_vm_path(**expected)
    actual = VmMigrationClient.parse_migrating_vm_path(path)
    assert expected == actual

def test_replication_cycle_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    source = 'winkle'
    migrating_vm = 'nautilus'
    replication_cycle = 'scallop'
    expected = 'projects/{project}/locations/{location}/sources/{source}/migratingVms/{migrating_vm}/replicationCycles/{replication_cycle}'.format(project=project, location=location, source=source, migrating_vm=migrating_vm, replication_cycle=replication_cycle)
    actual = VmMigrationClient.replication_cycle_path(project, location, source, migrating_vm, replication_cycle)
    assert expected == actual

def test_parse_replication_cycle_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'source': 'clam', 'migrating_vm': 'whelk', 'replication_cycle': 'octopus'}
    path = VmMigrationClient.replication_cycle_path(**expected)
    actual = VmMigrationClient.parse_replication_cycle_path(path)
    assert expected == actual

def test_source_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    source = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/sources/{source}'.format(project=project, location=location, source=source)
    actual = VmMigrationClient.source_path(project, location, source)
    assert expected == actual

def test_parse_source_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'location': 'winkle', 'source': 'nautilus'}
    path = VmMigrationClient.source_path(**expected)
    actual = VmMigrationClient.parse_source_path(path)
    assert expected == actual

def test_target_project_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    target_project = 'squid'
    expected = 'projects/{project}/locations/{location}/targetProjects/{target_project}'.format(project=project, location=location, target_project=target_project)
    actual = VmMigrationClient.target_project_path(project, location, target_project)
    assert expected == actual

def test_parse_target_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam', 'location': 'whelk', 'target_project': 'octopus'}
    path = VmMigrationClient.target_project_path(**expected)
    actual = VmMigrationClient.parse_target_project_path(path)
    assert expected == actual

def test_utilization_report_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    source = 'cuttlefish'
    utilization_report = 'mussel'
    expected = 'projects/{project}/locations/{location}/sources/{source}/utilizationReports/{utilization_report}'.format(project=project, location=location, source=source, utilization_report=utilization_report)
    actual = VmMigrationClient.utilization_report_path(project, location, source, utilization_report)
    assert expected == actual

def test_parse_utilization_report_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'location': 'nautilus', 'source': 'scallop', 'utilization_report': 'abalone'}
    path = VmMigrationClient.utilization_report_path(**expected)
    actual = VmMigrationClient.parse_utilization_report_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = VmMigrationClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = VmMigrationClient.common_billing_account_path(**expected)
    actual = VmMigrationClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = VmMigrationClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = VmMigrationClient.common_folder_path(**expected)
    actual = VmMigrationClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = VmMigrationClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = VmMigrationClient.common_organization_path(**expected)
    actual = VmMigrationClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = VmMigrationClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = VmMigrationClient.common_project_path(**expected)
    actual = VmMigrationClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = VmMigrationClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = VmMigrationClient.common_location_path(**expected)
    actual = VmMigrationClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.VmMigrationTransport, '_prep_wrapped_messages') as prep:
        client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.VmMigrationTransport, '_prep_wrapped_messages') as prep:
        transport_class = VmMigrationClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = VmMigrationAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = VmMigrationClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(VmMigrationClient, transports.VmMigrationGrpcTransport), (VmMigrationAsyncClient, transports.VmMigrationGrpcAsyncIOTransport)])
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