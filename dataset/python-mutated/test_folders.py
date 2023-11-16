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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
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
from google.cloud.resourcemanager_v3.services.folders import FoldersAsyncClient, FoldersClient, pagers, transports
from google.cloud.resourcemanager_v3.types import folders

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert FoldersClient._get_default_mtls_endpoint(None) is None
    assert FoldersClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert FoldersClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert FoldersClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert FoldersClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert FoldersClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(FoldersClient, 'grpc'), (FoldersAsyncClient, 'grpc_asyncio'), (FoldersClient, 'rest')])
def test_folders_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.FoldersGrpcTransport, 'grpc'), (transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.FoldersRestTransport, 'rest')])
def test_folders_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(FoldersClient, 'grpc'), (FoldersAsyncClient, 'grpc_asyncio'), (FoldersClient, 'rest')])
def test_folders_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

def test_folders_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = FoldersClient.get_transport_class()
    available_transports = [transports.FoldersGrpcTransport, transports.FoldersRestTransport]
    assert transport in available_transports
    transport = FoldersClient.get_transport_class('grpc')
    assert transport == transports.FoldersGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(FoldersClient, transports.FoldersGrpcTransport, 'grpc'), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio'), (FoldersClient, transports.FoldersRestTransport, 'rest')])
@mock.patch.object(FoldersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FoldersClient))
@mock.patch.object(FoldersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FoldersAsyncClient))
def test_folders_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(FoldersClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(FoldersClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(FoldersClient, transports.FoldersGrpcTransport, 'grpc', 'true'), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (FoldersClient, transports.FoldersGrpcTransport, 'grpc', 'false'), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (FoldersClient, transports.FoldersRestTransport, 'rest', 'true'), (FoldersClient, transports.FoldersRestTransport, 'rest', 'false')])
@mock.patch.object(FoldersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FoldersClient))
@mock.patch.object(FoldersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FoldersAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_folders_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [FoldersClient, FoldersAsyncClient])
@mock.patch.object(FoldersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FoldersClient))
@mock.patch.object(FoldersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FoldersAsyncClient))
def test_folders_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(FoldersClient, transports.FoldersGrpcTransport, 'grpc'), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio'), (FoldersClient, transports.FoldersRestTransport, 'rest')])
def test_folders_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(FoldersClient, transports.FoldersGrpcTransport, 'grpc', grpc_helpers), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (FoldersClient, transports.FoldersRestTransport, 'rest', None)])
def test_folders_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_folders_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.resourcemanager_v3.services.folders.transports.FoldersGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = FoldersClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(FoldersClient, transports.FoldersGrpcTransport, 'grpc', grpc_helpers), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_folders_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudresourcemanager.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=None, default_host='cloudresourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [folders.GetFolderRequest, dict])
def test_get_folder(request_type, transport: str='grpc'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        call.return_value = folders.Folder(name='name_value', parent='parent_value', display_name='display_name_value', state=folders.Folder.State.ACTIVE, etag='etag_value')
        response = client.get_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.GetFolderRequest()
    assert isinstance(response, folders.Folder)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.display_name == 'display_name_value'
    assert response.state == folders.Folder.State.ACTIVE
    assert response.etag == 'etag_value'

def test_get_folder_empty_call():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        client.get_folder()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.GetFolderRequest()

@pytest.mark.asyncio
async def test_get_folder_async(transport: str='grpc_asyncio', request_type=folders.GetFolderRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.Folder(name='name_value', parent='parent_value', display_name='display_name_value', state=folders.Folder.State.ACTIVE, etag='etag_value'))
        response = await client.get_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.GetFolderRequest()
    assert isinstance(response, folders.Folder)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.display_name == 'display_name_value'
    assert response.state == folders.Folder.State.ACTIVE
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_folder_async_from_dict():
    await test_get_folder_async(request_type=dict)

def test_get_folder_field_headers():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.GetFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        call.return_value = folders.Folder()
        client.get_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_folder_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.GetFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.Folder())
        await client.get_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_folder_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        call.return_value = folders.Folder()
        client.get_folder(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_folder_flattened_error():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_folder(folders.GetFolderRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_folder_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_folder), '__call__') as call:
        call.return_value = folders.Folder()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.Folder())
        response = await client.get_folder(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_folder_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_folder(folders.GetFolderRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [folders.ListFoldersRequest, dict])
def test_list_folders(request_type, transport: str='grpc'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        call.return_value = folders.ListFoldersResponse(next_page_token='next_page_token_value')
        response = client.list_folders(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.ListFoldersRequest()
    assert isinstance(response, pagers.ListFoldersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_folders_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        client.list_folders()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.ListFoldersRequest()

@pytest.mark.asyncio
async def test_list_folders_async(transport: str='grpc_asyncio', request_type=folders.ListFoldersRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.ListFoldersResponse(next_page_token='next_page_token_value'))
        response = await client.list_folders(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.ListFoldersRequest()
    assert isinstance(response, pagers.ListFoldersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_folders_async_from_dict():
    await test_list_folders_async(request_type=dict)

def test_list_folders_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        call.return_value = folders.ListFoldersResponse()
        client.list_folders(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_folders_flattened_error():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_folders(folders.ListFoldersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_folders_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        call.return_value = folders.ListFoldersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.ListFoldersResponse())
        response = await client.list_folders(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_folders_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_folders(folders.ListFoldersRequest(), parent='parent_value')

def test_list_folders_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        call.side_effect = (folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.ListFoldersResponse(folders=[], next_page_token='def'), folders.ListFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        metadata = ()
        pager = client.list_folders(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, folders.Folder) for i in results))

def test_list_folders_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_folders), '__call__') as call:
        call.side_effect = (folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.ListFoldersResponse(folders=[], next_page_token='def'), folders.ListFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        pages = list(client.list_folders(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_folders_async_pager():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_folders), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.ListFoldersResponse(folders=[], next_page_token='def'), folders.ListFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        async_pager = await client.list_folders(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, folders.Folder) for i in responses))

@pytest.mark.asyncio
async def test_list_folders_async_pages():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_folders), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.ListFoldersResponse(folders=[], next_page_token='def'), folders.ListFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_folders(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [folders.SearchFoldersRequest, dict])
def test_search_folders(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        call.return_value = folders.SearchFoldersResponse(next_page_token='next_page_token_value')
        response = client.search_folders(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.SearchFoldersRequest()
    assert isinstance(response, pagers.SearchFoldersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_folders_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        client.search_folders()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.SearchFoldersRequest()

@pytest.mark.asyncio
async def test_search_folders_async(transport: str='grpc_asyncio', request_type=folders.SearchFoldersRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.SearchFoldersResponse(next_page_token='next_page_token_value'))
        response = await client.search_folders(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.SearchFoldersRequest()
    assert isinstance(response, pagers.SearchFoldersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_folders_async_from_dict():
    await test_search_folders_async(request_type=dict)

def test_search_folders_flattened():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        call.return_value = folders.SearchFoldersResponse()
        client.search_folders(query='query_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

def test_search_folders_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_folders(folders.SearchFoldersRequest(), query='query_value')

@pytest.mark.asyncio
async def test_search_folders_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        call.return_value = folders.SearchFoldersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(folders.SearchFoldersResponse())
        response = await client.search_folders(query='query_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_folders_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_folders(folders.SearchFoldersRequest(), query='query_value')

def test_search_folders_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        call.side_effect = (folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.SearchFoldersResponse(folders=[], next_page_token='def'), folders.SearchFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        metadata = ()
        pager = client.search_folders(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, folders.Folder) for i in results))

def test_search_folders_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_folders), '__call__') as call:
        call.side_effect = (folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.SearchFoldersResponse(folders=[], next_page_token='def'), folders.SearchFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        pages = list(client.search_folders(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_folders_async_pager():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_folders), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.SearchFoldersResponse(folders=[], next_page_token='def'), folders.SearchFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        async_pager = await client.search_folders(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, folders.Folder) for i in responses))

@pytest.mark.asyncio
async def test_search_folders_async_pages():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_folders), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.SearchFoldersResponse(folders=[], next_page_token='def'), folders.SearchFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_folders(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [folders.CreateFolderRequest, dict])
def test_create_folder(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.CreateFolderRequest()
    assert isinstance(response, future.Future)

def test_create_folder_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_folder), '__call__') as call:
        client.create_folder()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.CreateFolderRequest()

@pytest.mark.asyncio
async def test_create_folder_async(transport: str='grpc_asyncio', request_type=folders.CreateFolderRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.CreateFolderRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_folder_async_from_dict():
    await test_create_folder_async(request_type=dict)

def test_create_folder_flattened():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_folder(folder=folders.Folder(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].folder
        mock_val = folders.Folder(name='name_value')
        assert arg == mock_val

def test_create_folder_flattened_error():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_folder(folders.CreateFolderRequest(), folder=folders.Folder(name='name_value'))

@pytest.mark.asyncio
async def test_create_folder_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_folder(folder=folders.Folder(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].folder
        mock_val = folders.Folder(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_folder_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_folder(folders.CreateFolderRequest(), folder=folders.Folder(name='name_value'))

@pytest.mark.parametrize('request_type', [folders.UpdateFolderRequest, dict])
def test_update_folder(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.UpdateFolderRequest()
    assert isinstance(response, future.Future)

def test_update_folder_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        client.update_folder()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.UpdateFolderRequest()

@pytest.mark.asyncio
async def test_update_folder_async(transport: str='grpc_asyncio', request_type=folders.UpdateFolderRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.UpdateFolderRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_folder_async_from_dict():
    await test_update_folder_async(request_type=dict)

def test_update_folder_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.UpdateFolderRequest()
    request.folder.name = 'name_value'
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'folder.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_folder_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.UpdateFolderRequest()
    request.folder.name = 'name_value'
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'folder.name=name_value') in kw['metadata']

def test_update_folder_flattened():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_folder(folder=folders.Folder(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].folder
        mock_val = folders.Folder(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_folder_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_folder(folders.UpdateFolderRequest(), folder=folders.Folder(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_folder_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_folder(folder=folders.Folder(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].folder
        mock_val = folders.Folder(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_folder_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_folder(folders.UpdateFolderRequest(), folder=folders.Folder(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [folders.MoveFolderRequest, dict])
def test_move_folder(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.move_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.MoveFolderRequest()
    assert isinstance(response, future.Future)

def test_move_folder_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        client.move_folder()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.MoveFolderRequest()

@pytest.mark.asyncio
async def test_move_folder_async(transport: str='grpc_asyncio', request_type=folders.MoveFolderRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.move_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.MoveFolderRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_move_folder_async_from_dict():
    await test_move_folder_async(request_type=dict)

def test_move_folder_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.MoveFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.move_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_move_folder_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.MoveFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.move_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_move_folder_flattened():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.move_folder(name='name_value', destination_parent='destination_parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].destination_parent
        mock_val = 'destination_parent_value'
        assert arg == mock_val

def test_move_folder_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.move_folder(folders.MoveFolderRequest(), name='name_value', destination_parent='destination_parent_value')

@pytest.mark.asyncio
async def test_move_folder_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.move_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.move_folder(name='name_value', destination_parent='destination_parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].destination_parent
        mock_val = 'destination_parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_move_folder_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.move_folder(folders.MoveFolderRequest(), name='name_value', destination_parent='destination_parent_value')

@pytest.mark.parametrize('request_type', [folders.DeleteFolderRequest, dict])
def test_delete_folder(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.DeleteFolderRequest()
    assert isinstance(response, future.Future)

def test_delete_folder_empty_call():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        client.delete_folder()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.DeleteFolderRequest()

@pytest.mark.asyncio
async def test_delete_folder_async(transport: str='grpc_asyncio', request_type=folders.DeleteFolderRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.DeleteFolderRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_folder_async_from_dict():
    await test_delete_folder_async(request_type=dict)

def test_delete_folder_field_headers():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.DeleteFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_folder_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.DeleteFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_folder_flattened():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_folder(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_folder_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_folder(folders.DeleteFolderRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_folder_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_folder(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_folder_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_folder(folders.DeleteFolderRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [folders.UndeleteFolderRequest, dict])
def test_undelete_folder(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.undelete_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.UndeleteFolderRequest()
    assert isinstance(response, future.Future)

def test_undelete_folder_empty_call():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        client.undelete_folder()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.UndeleteFolderRequest()

@pytest.mark.asyncio
async def test_undelete_folder_async(transport: str='grpc_asyncio', request_type=folders.UndeleteFolderRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undelete_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == folders.UndeleteFolderRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_undelete_folder_async_from_dict():
    await test_undelete_folder_async(request_type=dict)

def test_undelete_folder_field_headers():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.UndeleteFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undelete_folder(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_undelete_folder_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = folders.UndeleteFolderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.undelete_folder(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_undelete_folder_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undelete_folder(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_undelete_folder_flattened_error():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.undelete_folder(folders.UndeleteFolderRequest(), name='name_value')

@pytest.mark.asyncio
async def test_undelete_folder_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undelete_folder), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undelete_folder(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_undelete_folder_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.undelete_folder(folders.UndeleteFolderRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_get_iam_policy_flattened():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_get_iam_policy_flattened_error():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

def test_set_iam_policy_flattened():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_set_iam_policy_flattened_error():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_test_iam_permissions_flattened():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

def test_test_iam_permissions_flattened_error():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_error_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.parametrize('request_type', [folders.GetFolderRequest, dict])
def test_get_folder_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = folders.Folder(name='name_value', parent='parent_value', display_name='display_name_value', state=folders.Folder.State.ACTIVE, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = folders.Folder.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_folder(request)
    assert isinstance(response, folders.Folder)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.display_name == 'display_name_value'
    assert response.state == folders.Folder.State.ACTIVE
    assert response.etag == 'etag_value'

def test_get_folder_rest_required_fields(request_type=folders.GetFolderRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = folders.Folder()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = folders.Folder.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_folder(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_folder_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_folder._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_folder_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FoldersRestInterceptor, 'post_get_folder') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_get_folder') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.GetFolderRequest.pb(folders.GetFolderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = folders.Folder.to_json(folders.Folder())
        request = folders.GetFolderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = folders.Folder()
        client.get_folder(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_folder_rest_bad_request(transport: str='rest', request_type=folders.GetFolderRequest):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_folder(request)

def test_get_folder_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = folders.Folder()
        sample_request = {'name': 'folders/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = folders.Folder.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_folder(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=folders/*}' % client.transport._host, args[1])

def test_get_folder_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_folder(folders.GetFolderRequest(), name='name_value')

def test_get_folder_rest_error():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [folders.ListFoldersRequest, dict])
def test_list_folders_rest(request_type):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = folders.ListFoldersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = folders.ListFoldersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_folders(request)
    assert isinstance(response, pagers.ListFoldersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_folders_rest_required_fields(request_type=folders.ListFoldersRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'parent' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_folders._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == request_init['parent']
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_folders._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'parent', 'show_deleted'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = folders.ListFoldersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = folders.ListFoldersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_folders(request)
            expected_params = [('parent', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_folders_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_folders._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'parent', 'showDeleted')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_folders_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FoldersRestInterceptor, 'post_list_folders') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_list_folders') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.ListFoldersRequest.pb(folders.ListFoldersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = folders.ListFoldersResponse.to_json(folders.ListFoldersResponse())
        request = folders.ListFoldersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = folders.ListFoldersResponse()
        client.list_folders(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_folders_rest_bad_request(transport: str='rest', request_type=folders.ListFoldersRequest):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_folders(request)

def test_list_folders_rest_flattened():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = folders.ListFoldersResponse()
        sample_request = {}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = folders.ListFoldersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_folders(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/folders' % client.transport._host, args[1])

def test_list_folders_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_folders(folders.ListFoldersRequest(), parent='parent_value')

def test_list_folders_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.ListFoldersResponse(folders=[], next_page_token='def'), folders.ListFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.ListFoldersResponse(folders=[folders.Folder(), folders.Folder()]))
        response = response + response
        response = tuple((folders.ListFoldersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_folders(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, folders.Folder) for i in results))
        pages = list(client.list_folders(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [folders.SearchFoldersRequest, dict])
def test_search_folders_rest(request_type):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = folders.SearchFoldersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = folders.SearchFoldersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_folders(request)
    assert isinstance(response, pagers.SearchFoldersPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_folders_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FoldersRestInterceptor, 'post_search_folders') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_search_folders') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.SearchFoldersRequest.pb(folders.SearchFoldersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = folders.SearchFoldersResponse.to_json(folders.SearchFoldersResponse())
        request = folders.SearchFoldersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = folders.SearchFoldersResponse()
        client.search_folders(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_folders_rest_bad_request(transport: str='rest', request_type=folders.SearchFoldersRequest):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_folders(request)

def test_search_folders_rest_flattened():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = folders.SearchFoldersResponse()
        sample_request = {}
        mock_args = dict(query='query_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = folders.SearchFoldersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_folders(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/folders:search' % client.transport._host, args[1])

def test_search_folders_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_folders(folders.SearchFoldersRequest(), query='query_value')

def test_search_folders_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder(), folders.Folder()], next_page_token='abc'), folders.SearchFoldersResponse(folders=[], next_page_token='def'), folders.SearchFoldersResponse(folders=[folders.Folder()], next_page_token='ghi'), folders.SearchFoldersResponse(folders=[folders.Folder(), folders.Folder()]))
        response = response + response
        response = tuple((folders.SearchFoldersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.search_folders(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, folders.Folder) for i in results))
        pages = list(client.search_folders(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [folders.CreateFolderRequest, dict])
def test_create_folder_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request_init['folder'] = {'name': 'name_value', 'parent': 'parent_value', 'display_name': 'display_name_value', 'state': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'etag': 'etag_value'}
    test_field = folders.CreateFolderRequest.meta.fields['folder']

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
    for (field, value) in request_init['folder'].items():
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
                for i in range(0, len(request_init['folder'][field])):
                    del request_init['folder'][field][i][subfield]
            else:
                del request_init['folder'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_folder(request)
    assert response.operation.name == 'operations/spam'

def test_create_folder_rest_required_fields(request_type=folders.CreateFolderRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_folder(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_folder_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_folder._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('folder',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_folder_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FoldersRestInterceptor, 'post_create_folder') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_create_folder') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.CreateFolderRequest.pb(folders.CreateFolderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = folders.CreateFolderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_folder(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_folder_rest_bad_request(transport: str='rest', request_type=folders.CreateFolderRequest):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_folder(request)

def test_create_folder_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {}
        mock_args = dict(folder=folders.Folder(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_folder(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/folders' % client.transport._host, args[1])

def test_create_folder_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_folder(folders.CreateFolderRequest(), folder=folders.Folder(name='name_value'))

def test_create_folder_rest_error():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [folders.UpdateFolderRequest, dict])
def test_update_folder_rest(request_type):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'folder': {'name': 'folders/sample1'}}
    request_init['folder'] = {'name': 'folders/sample1', 'parent': 'parent_value', 'display_name': 'display_name_value', 'state': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'etag': 'etag_value'}
    test_field = folders.UpdateFolderRequest.meta.fields['folder']

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
    for (field, value) in request_init['folder'].items():
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
                for i in range(0, len(request_init['folder'][field])):
                    del request_init['folder'][field][i][subfield]
            else:
                del request_init['folder'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_folder(request)
    assert response.operation.name == 'operations/spam'

def test_update_folder_rest_required_fields(request_type=folders.UpdateFolderRequest):
    if False:
        return 10
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_folder._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_folder(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_folder_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_folder._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('folder', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_folder_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FoldersRestInterceptor, 'post_update_folder') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_update_folder') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.UpdateFolderRequest.pb(folders.UpdateFolderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = folders.UpdateFolderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_folder(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_folder_rest_bad_request(transport: str='rest', request_type=folders.UpdateFolderRequest):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'folder': {'name': 'folders/sample1'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_folder(request)

def test_update_folder_rest_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'folder': {'name': 'folders/sample1'}}
        mock_args = dict(folder=folders.Folder(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_folder(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{folder.name=folders/*}' % client.transport._host, args[1])

def test_update_folder_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_folder(folders.UpdateFolderRequest(), folder=folders.Folder(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_folder_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [folders.MoveFolderRequest, dict])
def test_move_folder_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.move_folder(request)
    assert response.operation.name == 'operations/spam'

def test_move_folder_rest_required_fields(request_type=folders.MoveFolderRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['destination_parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).move_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['destinationParent'] = 'destination_parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).move_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'destinationParent' in jsonified_request
    assert jsonified_request['destinationParent'] == 'destination_parent_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.move_folder(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_move_folder_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.move_folder._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'destinationParent'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_move_folder_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FoldersRestInterceptor, 'post_move_folder') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_move_folder') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.MoveFolderRequest.pb(folders.MoveFolderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = folders.MoveFolderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.move_folder(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_move_folder_rest_bad_request(transport: str='rest', request_type=folders.MoveFolderRequest):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.move_folder(request)

def test_move_folder_rest_flattened():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'folders/sample1'}
        mock_args = dict(name='name_value', destination_parent='destination_parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.move_folder(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=folders/*}:move' % client.transport._host, args[1])

def test_move_folder_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.move_folder(folders.MoveFolderRequest(), name='name_value', destination_parent='destination_parent_value')

def test_move_folder_rest_error():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [folders.DeleteFolderRequest, dict])
def test_delete_folder_rest(request_type):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_folder(request)
    assert response.operation.name == 'operations/spam'

def test_delete_folder_rest_required_fields(request_type=folders.DeleteFolderRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_folder(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_folder_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_folder._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_folder_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FoldersRestInterceptor, 'post_delete_folder') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_delete_folder') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.DeleteFolderRequest.pb(folders.DeleteFolderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = folders.DeleteFolderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_folder(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_folder_rest_bad_request(transport: str='rest', request_type=folders.DeleteFolderRequest):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_folder(request)

def test_delete_folder_rest_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'folders/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_folder(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=folders/*}' % client.transport._host, args[1])

def test_delete_folder_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_folder(folders.DeleteFolderRequest(), name='name_value')

def test_delete_folder_rest_error():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [folders.UndeleteFolderRequest, dict])
def test_undelete_folder_rest(request_type):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.undelete_folder(request)
    assert response.operation.name == 'operations/spam'

def test_undelete_folder_rest_required_fields(request_type=folders.UndeleteFolderRequest):
    if False:
        print('Hello World!')
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undelete_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undelete_folder._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.undelete_folder(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_undelete_folder_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.undelete_folder._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_undelete_folder_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FoldersRestInterceptor, 'post_undelete_folder') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_undelete_folder') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = folders.UndeleteFolderRequest.pb(folders.UndeleteFolderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = folders.UndeleteFolderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.undelete_folder(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_undelete_folder_rest_bad_request(transport: str='rest', request_type=folders.UndeleteFolderRequest):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.undelete_folder(request)

def test_undelete_folder_rest_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'folders/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.undelete_folder(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=folders/*}:undelete' % client.transport._host, args[1])

def test_undelete_folder_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.undelete_folder(folders.UndeleteFolderRequest(), name='name_value')

def test_undelete_folder_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FoldersRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_get_iam_policy') as pre:
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
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_flattened():
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'folders/sample1'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{resource=folders/*}:getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FoldersRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_set_iam_policy') as pre:
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
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'folders/sample1'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{resource=folders/*}:setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

def test_set_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        print('Hello World!')
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.FoldersRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.FoldersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FoldersRestInterceptor())
    client = FoldersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FoldersRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.FoldersRestInterceptor, 'pre_test_iam_permissions') as pre:
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
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'folders/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        sample_request = {'resource': 'folders/sample1'}
        mock_args = dict(resource='resource_value', permissions=['permissions_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{resource=folders/*}:testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

def test_test_iam_permissions_rest_error():
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FoldersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.FoldersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = FoldersClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.FoldersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = FoldersClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = FoldersClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.FoldersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = FoldersClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.FoldersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = FoldersClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.FoldersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.FoldersGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.FoldersGrpcTransport, transports.FoldersGrpcAsyncIOTransport, transports.FoldersRestTransport])
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
        for i in range(10):
            print('nop')
    transport = FoldersClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.FoldersGrpcTransport)

def test_folders_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.FoldersTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_folders_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.resourcemanager_v3.services.folders.transports.FoldersTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.FoldersTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_folder', 'list_folders', 'search_folders', 'create_folder', 'update_folder', 'move_folder', 'delete_folder', 'undelete_folder', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions', 'get_operation')
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

def test_folders_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.resourcemanager_v3.services.folders.transports.FoldersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.FoldersTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

def test_folders_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.resourcemanager_v3.services.folders.transports.FoldersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.FoldersTransport()
        adc.assert_called_once()

def test_folders_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        FoldersClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.FoldersGrpcTransport, transports.FoldersGrpcAsyncIOTransport])
def test_folders_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.FoldersGrpcTransport, transports.FoldersGrpcAsyncIOTransport, transports.FoldersRestTransport])
def test_folders_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.FoldersGrpcTransport, grpc_helpers), (transports.FoldersGrpcAsyncIOTransport, grpc_helpers_async)])
def test_folders_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudresourcemanager.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=['1', '2'], default_host='cloudresourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.FoldersGrpcTransport, transports.FoldersGrpcAsyncIOTransport])
def test_folders_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_folders_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.FoldersRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_folders_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_folders_host_no_port(transport_name):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudresourcemanager.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_folders_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudresourcemanager.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudresourcemanager.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_folders_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = FoldersClient(credentials=creds1, transport=transport_name)
    client2 = FoldersClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_folder._session
    session2 = client2.transport.get_folder._session
    assert session1 != session2
    session1 = client1.transport.list_folders._session
    session2 = client2.transport.list_folders._session
    assert session1 != session2
    session1 = client1.transport.search_folders._session
    session2 = client2.transport.search_folders._session
    assert session1 != session2
    session1 = client1.transport.create_folder._session
    session2 = client2.transport.create_folder._session
    assert session1 != session2
    session1 = client1.transport.update_folder._session
    session2 = client2.transport.update_folder._session
    assert session1 != session2
    session1 = client1.transport.move_folder._session
    session2 = client2.transport.move_folder._session
    assert session1 != session2
    session1 = client1.transport.delete_folder._session
    session2 = client2.transport.delete_folder._session
    assert session1 != session2
    session1 = client1.transport.undelete_folder._session
    session2 = client2.transport.undelete_folder._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_folders_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.FoldersGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_folders_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.FoldersGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.FoldersGrpcTransport, transports.FoldersGrpcAsyncIOTransport])
def test_folders_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.FoldersGrpcTransport, transports.FoldersGrpcAsyncIOTransport])
def test_folders_transport_channel_mtls_with_adc(transport_class):
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

def test_folders_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_folders_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_folder_path():
    if False:
        while True:
            i = 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = FoldersClient.folder_path(folder)
    assert expected == actual

def test_parse_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'clam'}
    path = FoldersClient.folder_path(**expected)
    actual = FoldersClient.parse_folder_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = FoldersClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = FoldersClient.common_billing_account_path(**expected)
    actual = FoldersClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = FoldersClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nudibranch'}
    path = FoldersClient.common_folder_path(**expected)
    actual = FoldersClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = FoldersClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'mussel'}
    path = FoldersClient.common_organization_path(**expected)
    actual = FoldersClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = FoldersClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus'}
    path = FoldersClient.common_project_path(**expected)
    actual = FoldersClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = FoldersClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam'}
    path = FoldersClient.common_location_path(**expected)
    actual = FoldersClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.FoldersTransport, '_prep_wrapped_messages') as prep:
        client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.FoldersTransport, '_prep_wrapped_messages') as prep:
        transport_class = FoldersClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = FoldersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = FoldersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = FoldersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(FoldersClient, transports.FoldersGrpcTransport), (FoldersAsyncClient, transports.FoldersGrpcAsyncIOTransport)])
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