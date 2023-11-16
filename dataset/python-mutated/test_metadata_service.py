import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
import math
from google.api_core import gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.cloud.location import locations_pb2
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.dataplex_v1.services.metadata_service import MetadataServiceAsyncClient, MetadataServiceClient, pagers, transports
from google.cloud.dataplex_v1.types import metadata_

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert MetadataServiceClient._get_default_mtls_endpoint(None) is None
    assert MetadataServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert MetadataServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert MetadataServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert MetadataServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert MetadataServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(MetadataServiceClient, 'grpc'), (MetadataServiceAsyncClient, 'grpc_asyncio')])
def test_metadata_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'dataplex.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.MetadataServiceGrpcTransport, 'grpc'), (transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_metadata_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(MetadataServiceClient, 'grpc'), (MetadataServiceAsyncClient, 'grpc_asyncio')])
def test_metadata_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'dataplex.googleapis.com:443'

def test_metadata_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = MetadataServiceClient.get_transport_class()
    available_transports = [transports.MetadataServiceGrpcTransport]
    assert transport in available_transports
    transport = MetadataServiceClient.get_transport_class('grpc')
    assert transport == transports.MetadataServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(MetadataServiceClient, transports.MetadataServiceGrpcTransport, 'grpc'), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(MetadataServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetadataServiceClient))
@mock.patch.object(MetadataServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetadataServiceAsyncClient))
def test_metadata_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(MetadataServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(MetadataServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(MetadataServiceClient, transports.MetadataServiceGrpcTransport, 'grpc', 'true'), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (MetadataServiceClient, transports.MetadataServiceGrpcTransport, 'grpc', 'false'), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(MetadataServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetadataServiceClient))
@mock.patch.object(MetadataServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetadataServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_metadata_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class', [MetadataServiceClient, MetadataServiceAsyncClient])
@mock.patch.object(MetadataServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetadataServiceClient))
@mock.patch.object(MetadataServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetadataServiceAsyncClient))
def test_metadata_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(MetadataServiceClient, transports.MetadataServiceGrpcTransport, 'grpc'), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_metadata_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(MetadataServiceClient, transports.MetadataServiceGrpcTransport, 'grpc', grpc_helpers), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_metadata_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_metadata_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dataplex_v1.services.metadata_service.transports.MetadataServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = MetadataServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(MetadataServiceClient, transports.MetadataServiceGrpcTransport, 'grpc', grpc_helpers), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_metadata_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dataplex.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='dataplex.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [metadata_.CreateEntityRequest, dict])
def test_create_entity(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        call.return_value = metadata_.Entity(name='name_value', display_name='display_name_value', description='description_value', id='id_value', etag='etag_value', type_=metadata_.Entity.Type.TABLE, asset='asset_value', data_path='data_path_value', data_path_pattern='data_path_pattern_value', catalog_entry='catalog_entry_value', system=metadata_.StorageSystem.CLOUD_STORAGE, uid='uid_value')
        response = client.create_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.CreateEntityRequest()
    assert isinstance(response, metadata_.Entity)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.id == 'id_value'
    assert response.etag == 'etag_value'
    assert response.type_ == metadata_.Entity.Type.TABLE
    assert response.asset == 'asset_value'
    assert response.data_path == 'data_path_value'
    assert response.data_path_pattern == 'data_path_pattern_value'
    assert response.catalog_entry == 'catalog_entry_value'
    assert response.system == metadata_.StorageSystem.CLOUD_STORAGE
    assert response.uid == 'uid_value'

def test_create_entity_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        client.create_entity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.CreateEntityRequest()

@pytest.mark.asyncio
async def test_create_entity_async(transport: str='grpc_asyncio', request_type=metadata_.CreateEntityRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity(name='name_value', display_name='display_name_value', description='description_value', id='id_value', etag='etag_value', type_=metadata_.Entity.Type.TABLE, asset='asset_value', data_path='data_path_value', data_path_pattern='data_path_pattern_value', catalog_entry='catalog_entry_value', system=metadata_.StorageSystem.CLOUD_STORAGE, uid='uid_value'))
        response = await client.create_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.CreateEntityRequest()
    assert isinstance(response, metadata_.Entity)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.id == 'id_value'
    assert response.etag == 'etag_value'
    assert response.type_ == metadata_.Entity.Type.TABLE
    assert response.asset == 'asset_value'
    assert response.data_path == 'data_path_value'
    assert response.data_path_pattern == 'data_path_pattern_value'
    assert response.catalog_entry == 'catalog_entry_value'
    assert response.system == metadata_.StorageSystem.CLOUD_STORAGE
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_create_entity_async_from_dict():
    await test_create_entity_async(request_type=dict)

def test_create_entity_field_headers():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.CreateEntityRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        client.create_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_entity_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.CreateEntityRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity())
        await client.create_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_entity_flattened():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        client.create_entity(parent='parent_value', entity=metadata_.Entity(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity
        mock_val = metadata_.Entity(name='name_value')
        assert arg == mock_val

def test_create_entity_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_entity(metadata_.CreateEntityRequest(), parent='parent_value', entity=metadata_.Entity(name='name_value'))

@pytest.mark.asyncio
async def test_create_entity_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity())
        response = await client.create_entity(parent='parent_value', entity=metadata_.Entity(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity
        mock_val = metadata_.Entity(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_entity_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_entity(metadata_.CreateEntityRequest(), parent='parent_value', entity=metadata_.Entity(name='name_value'))

@pytest.mark.parametrize('request_type', [metadata_.UpdateEntityRequest, dict])
def test_update_entity(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_entity), '__call__') as call:
        call.return_value = metadata_.Entity(name='name_value', display_name='display_name_value', description='description_value', id='id_value', etag='etag_value', type_=metadata_.Entity.Type.TABLE, asset='asset_value', data_path='data_path_value', data_path_pattern='data_path_pattern_value', catalog_entry='catalog_entry_value', system=metadata_.StorageSystem.CLOUD_STORAGE, uid='uid_value')
        response = client.update_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.UpdateEntityRequest()
    assert isinstance(response, metadata_.Entity)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.id == 'id_value'
    assert response.etag == 'etag_value'
    assert response.type_ == metadata_.Entity.Type.TABLE
    assert response.asset == 'asset_value'
    assert response.data_path == 'data_path_value'
    assert response.data_path_pattern == 'data_path_pattern_value'
    assert response.catalog_entry == 'catalog_entry_value'
    assert response.system == metadata_.StorageSystem.CLOUD_STORAGE
    assert response.uid == 'uid_value'

def test_update_entity_empty_call():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_entity), '__call__') as call:
        client.update_entity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.UpdateEntityRequest()

@pytest.mark.asyncio
async def test_update_entity_async(transport: str='grpc_asyncio', request_type=metadata_.UpdateEntityRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity(name='name_value', display_name='display_name_value', description='description_value', id='id_value', etag='etag_value', type_=metadata_.Entity.Type.TABLE, asset='asset_value', data_path='data_path_value', data_path_pattern='data_path_pattern_value', catalog_entry='catalog_entry_value', system=metadata_.StorageSystem.CLOUD_STORAGE, uid='uid_value'))
        response = await client.update_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.UpdateEntityRequest()
    assert isinstance(response, metadata_.Entity)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.id == 'id_value'
    assert response.etag == 'etag_value'
    assert response.type_ == metadata_.Entity.Type.TABLE
    assert response.asset == 'asset_value'
    assert response.data_path == 'data_path_value'
    assert response.data_path_pattern == 'data_path_pattern_value'
    assert response.catalog_entry == 'catalog_entry_value'
    assert response.system == metadata_.StorageSystem.CLOUD_STORAGE
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_update_entity_async_from_dict():
    await test_update_entity_async(request_type=dict)

def test_update_entity_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.UpdateEntityRequest()
    request.entity.name = 'name_value'
    with mock.patch.object(type(client.transport.update_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        client.update_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'entity.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_entity_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.UpdateEntityRequest()
    request.entity.name = 'name_value'
    with mock.patch.object(type(client.transport.update_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity())
        await client.update_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'entity.name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [metadata_.DeleteEntityRequest, dict])
def test_delete_entity(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        call.return_value = None
        response = client.delete_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.DeleteEntityRequest()
    assert response is None

def test_delete_entity_empty_call():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        client.delete_entity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.DeleteEntityRequest()

@pytest.mark.asyncio
async def test_delete_entity_async(transport: str='grpc_asyncio', request_type=metadata_.DeleteEntityRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.DeleteEntityRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_entity_async_from_dict():
    await test_delete_entity_async(request_type=dict)

def test_delete_entity_field_headers():
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.DeleteEntityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        call.return_value = None
        client.delete_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_entity_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.DeleteEntityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_entity_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        call.return_value = None
        client.delete_entity(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_entity_flattened_error():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_entity(metadata_.DeleteEntityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_entity_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_entity), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_entity(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_entity_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_entity(metadata_.DeleteEntityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metadata_.GetEntityRequest, dict])
def test_get_entity(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        call.return_value = metadata_.Entity(name='name_value', display_name='display_name_value', description='description_value', id='id_value', etag='etag_value', type_=metadata_.Entity.Type.TABLE, asset='asset_value', data_path='data_path_value', data_path_pattern='data_path_pattern_value', catalog_entry='catalog_entry_value', system=metadata_.StorageSystem.CLOUD_STORAGE, uid='uid_value')
        response = client.get_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.GetEntityRequest()
    assert isinstance(response, metadata_.Entity)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.id == 'id_value'
    assert response.etag == 'etag_value'
    assert response.type_ == metadata_.Entity.Type.TABLE
    assert response.asset == 'asset_value'
    assert response.data_path == 'data_path_value'
    assert response.data_path_pattern == 'data_path_pattern_value'
    assert response.catalog_entry == 'catalog_entry_value'
    assert response.system == metadata_.StorageSystem.CLOUD_STORAGE
    assert response.uid == 'uid_value'

def test_get_entity_empty_call():
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        client.get_entity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.GetEntityRequest()

@pytest.mark.asyncio
async def test_get_entity_async(transport: str='grpc_asyncio', request_type=metadata_.GetEntityRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity(name='name_value', display_name='display_name_value', description='description_value', id='id_value', etag='etag_value', type_=metadata_.Entity.Type.TABLE, asset='asset_value', data_path='data_path_value', data_path_pattern='data_path_pattern_value', catalog_entry='catalog_entry_value', system=metadata_.StorageSystem.CLOUD_STORAGE, uid='uid_value'))
        response = await client.get_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.GetEntityRequest()
    assert isinstance(response, metadata_.Entity)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.id == 'id_value'
    assert response.etag == 'etag_value'
    assert response.type_ == metadata_.Entity.Type.TABLE
    assert response.asset == 'asset_value'
    assert response.data_path == 'data_path_value'
    assert response.data_path_pattern == 'data_path_pattern_value'
    assert response.catalog_entry == 'catalog_entry_value'
    assert response.system == metadata_.StorageSystem.CLOUD_STORAGE
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_entity_async_from_dict():
    await test_get_entity_async(request_type=dict)

def test_get_entity_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.GetEntityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        client.get_entity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_entity_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.GetEntityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity())
        await client.get_entity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_entity_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        client.get_entity(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_entity_flattened_error():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_entity(metadata_.GetEntityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_entity_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_entity), '__call__') as call:
        call.return_value = metadata_.Entity()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Entity())
        response = await client.get_entity(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_entity_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_entity(metadata_.GetEntityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metadata_.ListEntitiesRequest, dict])
def test_list_entities(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.return_value = metadata_.ListEntitiesResponse(next_page_token='next_page_token_value')
        response = client.list_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.ListEntitiesRequest()
    assert isinstance(response, pagers.ListEntitiesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entities_empty_call():
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        client.list_entities()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.ListEntitiesRequest()

@pytest.mark.asyncio
async def test_list_entities_async(transport: str='grpc_asyncio', request_type=metadata_.ListEntitiesRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.ListEntitiesResponse(next_page_token='next_page_token_value'))
        response = await client.list_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.ListEntitiesRequest()
    assert isinstance(response, pagers.ListEntitiesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_entities_async_from_dict():
    await test_list_entities_async(request_type=dict)

def test_list_entities_field_headers():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.ListEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.return_value = metadata_.ListEntitiesResponse()
        client.list_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_entities_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.ListEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.ListEntitiesResponse())
        await client.list_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_entities_flattened():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.return_value = metadata_.ListEntitiesResponse()
        client.list_entities(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_entities_flattened_error():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_entities(metadata_.ListEntitiesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_entities_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.return_value = metadata_.ListEntitiesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.ListEntitiesResponse())
        response = await client.list_entities(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_entities_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_entities(metadata_.ListEntitiesRequest(), parent='parent_value')

def test_list_entities_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.side_effect = (metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity(), metadata_.Entity()], next_page_token='abc'), metadata_.ListEntitiesResponse(entities=[], next_page_token='def'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity()], next_page_token='ghi'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_entities(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metadata_.Entity) for i in results))

def test_list_entities_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entities), '__call__') as call:
        call.side_effect = (metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity(), metadata_.Entity()], next_page_token='abc'), metadata_.ListEntitiesResponse(entities=[], next_page_token='def'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity()], next_page_token='ghi'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity()]), RuntimeError)
        pages = list(client.list_entities(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_entities_async_pager():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entities), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity(), metadata_.Entity()], next_page_token='abc'), metadata_.ListEntitiesResponse(entities=[], next_page_token='def'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity()], next_page_token='ghi'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity()]), RuntimeError)
        async_pager = await client.list_entities(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, metadata_.Entity) for i in responses))

@pytest.mark.asyncio
async def test_list_entities_async_pages():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entities), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity(), metadata_.Entity()], next_page_token='abc'), metadata_.ListEntitiesResponse(entities=[], next_page_token='def'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity()], next_page_token='ghi'), metadata_.ListEntitiesResponse(entities=[metadata_.Entity(), metadata_.Entity()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_entities(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metadata_.CreatePartitionRequest, dict])
def test_create_partition(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        call.return_value = metadata_.Partition(name='name_value', values=['values_value'], location='location_value', etag='etag_value')
        response = client.create_partition(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.CreatePartitionRequest()
    assert isinstance(response, metadata_.Partition)
    assert response.name == 'name_value'
    assert response.values == ['values_value']
    assert response.location == 'location_value'
    assert response.etag == 'etag_value'

def test_create_partition_empty_call():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        client.create_partition()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.CreatePartitionRequest()

@pytest.mark.asyncio
async def test_create_partition_async(transport: str='grpc_asyncio', request_type=metadata_.CreatePartitionRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Partition(name='name_value', values=['values_value'], location='location_value', etag='etag_value'))
        response = await client.create_partition(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.CreatePartitionRequest()
    assert isinstance(response, metadata_.Partition)
    assert response.name == 'name_value'
    assert response.values == ['values_value']
    assert response.location == 'location_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_create_partition_async_from_dict():
    await test_create_partition_async(request_type=dict)

def test_create_partition_field_headers():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.CreatePartitionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        call.return_value = metadata_.Partition()
        client.create_partition(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_partition_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.CreatePartitionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Partition())
        await client.create_partition(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_partition_flattened():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        call.return_value = metadata_.Partition()
        client.create_partition(parent='parent_value', partition=metadata_.Partition(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].partition
        mock_val = metadata_.Partition(name='name_value')
        assert arg == mock_val

def test_create_partition_flattened_error():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_partition(metadata_.CreatePartitionRequest(), parent='parent_value', partition=metadata_.Partition(name='name_value'))

@pytest.mark.asyncio
async def test_create_partition_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_partition), '__call__') as call:
        call.return_value = metadata_.Partition()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Partition())
        response = await client.create_partition(parent='parent_value', partition=metadata_.Partition(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].partition
        mock_val = metadata_.Partition(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_partition_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_partition(metadata_.CreatePartitionRequest(), parent='parent_value', partition=metadata_.Partition(name='name_value'))

@pytest.mark.parametrize('request_type', [metadata_.DeletePartitionRequest, dict])
def test_delete_partition(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        call.return_value = None
        response = client.delete_partition(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.DeletePartitionRequest()
    assert response is None

def test_delete_partition_empty_call():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        client.delete_partition()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.DeletePartitionRequest()

@pytest.mark.asyncio
async def test_delete_partition_async(transport: str='grpc_asyncio', request_type=metadata_.DeletePartitionRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_partition(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.DeletePartitionRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_partition_async_from_dict():
    await test_delete_partition_async(request_type=dict)

def test_delete_partition_field_headers():
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.DeletePartitionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        call.return_value = None
        client.delete_partition(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_partition_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.DeletePartitionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_partition(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_partition_flattened():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        call.return_value = None
        client.delete_partition(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_partition_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_partition(metadata_.DeletePartitionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_partition_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_partition), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_partition(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_partition_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_partition(metadata_.DeletePartitionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metadata_.GetPartitionRequest, dict])
def test_get_partition(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        call.return_value = metadata_.Partition(name='name_value', values=['values_value'], location='location_value', etag='etag_value')
        response = client.get_partition(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.GetPartitionRequest()
    assert isinstance(response, metadata_.Partition)
    assert response.name == 'name_value'
    assert response.values == ['values_value']
    assert response.location == 'location_value'
    assert response.etag == 'etag_value'

def test_get_partition_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        client.get_partition()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.GetPartitionRequest()

@pytest.mark.asyncio
async def test_get_partition_async(transport: str='grpc_asyncio', request_type=metadata_.GetPartitionRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Partition(name='name_value', values=['values_value'], location='location_value', etag='etag_value'))
        response = await client.get_partition(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.GetPartitionRequest()
    assert isinstance(response, metadata_.Partition)
    assert response.name == 'name_value'
    assert response.values == ['values_value']
    assert response.location == 'location_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_partition_async_from_dict():
    await test_get_partition_async(request_type=dict)

def test_get_partition_field_headers():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.GetPartitionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        call.return_value = metadata_.Partition()
        client.get_partition(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_partition_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.GetPartitionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Partition())
        await client.get_partition(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_partition_flattened():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        call.return_value = metadata_.Partition()
        client.get_partition(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_partition_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_partition(metadata_.GetPartitionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_partition_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_partition), '__call__') as call:
        call.return_value = metadata_.Partition()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.Partition())
        response = await client.get_partition(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_partition_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_partition(metadata_.GetPartitionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metadata_.ListPartitionsRequest, dict])
def test_list_partitions(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.return_value = metadata_.ListPartitionsResponse(next_page_token='next_page_token_value')
        response = client.list_partitions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.ListPartitionsRequest()
    assert isinstance(response, pagers.ListPartitionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_partitions_empty_call():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        client.list_partitions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.ListPartitionsRequest()

@pytest.mark.asyncio
async def test_list_partitions_async(transport: str='grpc_asyncio', request_type=metadata_.ListPartitionsRequest):
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.ListPartitionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_partitions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metadata_.ListPartitionsRequest()
    assert isinstance(response, pagers.ListPartitionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_partitions_async_from_dict():
    await test_list_partitions_async(request_type=dict)

def test_list_partitions_field_headers():
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.ListPartitionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.return_value = metadata_.ListPartitionsResponse()
        client.list_partitions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_partitions_field_headers_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metadata_.ListPartitionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.ListPartitionsResponse())
        await client.list_partitions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_partitions_flattened():
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.return_value = metadata_.ListPartitionsResponse()
        client.list_partitions(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_partitions_flattened_error():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_partitions(metadata_.ListPartitionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_partitions_flattened_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.return_value = metadata_.ListPartitionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metadata_.ListPartitionsResponse())
        response = await client.list_partitions(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_partitions_flattened_error_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_partitions(metadata_.ListPartitionsRequest(), parent='parent_value')

def test_list_partitions_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.side_effect = (metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition(), metadata_.Partition()], next_page_token='abc'), metadata_.ListPartitionsResponse(partitions=[], next_page_token='def'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition()], next_page_token='ghi'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_partitions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metadata_.Partition) for i in results))

def test_list_partitions_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_partitions), '__call__') as call:
        call.side_effect = (metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition(), metadata_.Partition()], next_page_token='abc'), metadata_.ListPartitionsResponse(partitions=[], next_page_token='def'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition()], next_page_token='ghi'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition()]), RuntimeError)
        pages = list(client.list_partitions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_partitions_async_pager():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_partitions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition(), metadata_.Partition()], next_page_token='abc'), metadata_.ListPartitionsResponse(partitions=[], next_page_token='def'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition()], next_page_token='ghi'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition()]), RuntimeError)
        async_pager = await client.list_partitions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, metadata_.Partition) for i in responses))

@pytest.mark.asyncio
async def test_list_partitions_async_pages():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_partitions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition(), metadata_.Partition()], next_page_token='abc'), metadata_.ListPartitionsResponse(partitions=[], next_page_token='def'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition()], next_page_token='ghi'), metadata_.ListPartitionsResponse(partitions=[metadata_.Partition(), metadata_.Partition()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_partitions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetadataServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.MetadataServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MetadataServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.MetadataServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = MetadataServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = MetadataServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.MetadataServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MetadataServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.MetadataServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = MetadataServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.MetadataServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.MetadataServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.MetadataServiceGrpcTransport, transports.MetadataServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = MetadataServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.MetadataServiceGrpcTransport)

def test_metadata_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.MetadataServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_metadata_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dataplex_v1.services.metadata_service.transports.MetadataServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.MetadataServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_entity', 'update_entity', 'delete_entity', 'get_entity', 'list_entities', 'create_partition', 'delete_partition', 'get_partition', 'list_partitions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_metadata_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataplex_v1.services.metadata_service.transports.MetadataServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.MetadataServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_metadata_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataplex_v1.services.metadata_service.transports.MetadataServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.MetadataServiceTransport()
        adc.assert_called_once()

def test_metadata_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        MetadataServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.MetadataServiceGrpcTransport, transports.MetadataServiceGrpcAsyncIOTransport])
def test_metadata_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.MetadataServiceGrpcTransport, transports.MetadataServiceGrpcAsyncIOTransport])
def test_metadata_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.MetadataServiceGrpcTransport, grpc_helpers), (transports.MetadataServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_metadata_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataplex.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dataplex.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.MetadataServiceGrpcTransport, transports.MetadataServiceGrpcAsyncIOTransport])
def test_metadata_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_metadata_service_host_no_port(transport_name):
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataplex.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'dataplex.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_metadata_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataplex.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'dataplex.googleapis.com:8000'

def test_metadata_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.MetadataServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_metadata_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.MetadataServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.MetadataServiceGrpcTransport, transports.MetadataServiceGrpcAsyncIOTransport])
def test_metadata_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.MetadataServiceGrpcTransport, transports.MetadataServiceGrpcAsyncIOTransport])
def test_metadata_service_transport_channel_mtls_with_adc(transport_class):
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

def test_entity_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    lake = 'whelk'
    zone = 'octopus'
    entity = 'oyster'
    expected = 'projects/{project}/locations/{location}/lakes/{lake}/zones/{zone}/entities/{entity}'.format(project=project, location=location, lake=lake, zone=zone, entity=entity)
    actual = MetadataServiceClient.entity_path(project, location, lake, zone, entity)
    assert expected == actual

def test_parse_entity_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'lake': 'mussel', 'zone': 'winkle', 'entity': 'nautilus'}
    path = MetadataServiceClient.entity_path(**expected)
    actual = MetadataServiceClient.parse_entity_path(path)
    assert expected == actual

def test_partition_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    lake = 'squid'
    zone = 'clam'
    entity = 'whelk'
    partition = 'octopus'
    expected = 'projects/{project}/locations/{location}/lakes/{lake}/zones/{zone}/entities/{entity}/partitions/{partition}'.format(project=project, location=location, lake=lake, zone=zone, entity=entity, partition=partition)
    actual = MetadataServiceClient.partition_path(project, location, lake, zone, entity, partition)
    assert expected == actual

def test_parse_partition_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch', 'lake': 'cuttlefish', 'zone': 'mussel', 'entity': 'winkle', 'partition': 'nautilus'}
    path = MetadataServiceClient.partition_path(**expected)
    actual = MetadataServiceClient.parse_partition_path(path)
    assert expected == actual

def test_zone_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    lake = 'squid'
    zone = 'clam'
    expected = 'projects/{project}/locations/{location}/lakes/{lake}/zones/{zone}'.format(project=project, location=location, lake=lake, zone=zone)
    actual = MetadataServiceClient.zone_path(project, location, lake, zone)
    assert expected == actual

def test_parse_zone_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'location': 'octopus', 'lake': 'oyster', 'zone': 'nudibranch'}
    path = MetadataServiceClient.zone_path(**expected)
    actual = MetadataServiceClient.parse_zone_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = MetadataServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = MetadataServiceClient.common_billing_account_path(**expected)
    actual = MetadataServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = MetadataServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nautilus'}
    path = MetadataServiceClient.common_folder_path(**expected)
    actual = MetadataServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = MetadataServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'abalone'}
    path = MetadataServiceClient.common_organization_path(**expected)
    actual = MetadataServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = MetadataServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = MetadataServiceClient.common_project_path(**expected)
    actual = MetadataServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = MetadataServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = MetadataServiceClient.common_location_path(**expected)
    actual = MetadataServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.MetadataServiceTransport, '_prep_wrapped_messages') as prep:
        client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.MetadataServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = MetadataServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = MetadataServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        i = 10
        return i + 15
    transports = ['grpc']
    for transport in transports:
        client = MetadataServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(MetadataServiceClient, transports.MetadataServiceGrpcTransport), (MetadataServiceAsyncClient, transports.MetadataServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)