import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
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
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.dataplex_v1.services.data_taxonomy_service import DataTaxonomyServiceAsyncClient, DataTaxonomyServiceClient, pagers, transports
from google.cloud.dataplex_v1.types import data_taxonomy
from google.cloud.dataplex_v1.types import data_taxonomy as gcd_data_taxonomy
from google.cloud.dataplex_v1.types import security, service

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
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
    assert DataTaxonomyServiceClient._get_default_mtls_endpoint(None) is None
    assert DataTaxonomyServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DataTaxonomyServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DataTaxonomyServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DataTaxonomyServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DataTaxonomyServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DataTaxonomyServiceClient, 'grpc'), (DataTaxonomyServiceAsyncClient, 'grpc_asyncio')])
def test_data_taxonomy_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'dataplex.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DataTaxonomyServiceGrpcTransport, 'grpc'), (transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_data_taxonomy_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DataTaxonomyServiceClient, 'grpc'), (DataTaxonomyServiceAsyncClient, 'grpc_asyncio')])
def test_data_taxonomy_service_client_from_service_account_file(client_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
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

def test_data_taxonomy_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = DataTaxonomyServiceClient.get_transport_class()
    available_transports = [transports.DataTaxonomyServiceGrpcTransport]
    assert transport in available_transports
    transport = DataTaxonomyServiceClient.get_transport_class('grpc')
    assert transport == transports.DataTaxonomyServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport, 'grpc'), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(DataTaxonomyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTaxonomyServiceClient))
@mock.patch.object(DataTaxonomyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTaxonomyServiceAsyncClient))
def test_data_taxonomy_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(DataTaxonomyServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DataTaxonomyServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport, 'grpc', 'true'), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport, 'grpc', 'false'), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(DataTaxonomyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTaxonomyServiceClient))
@mock.patch.object(DataTaxonomyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTaxonomyServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_data_taxonomy_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [DataTaxonomyServiceClient, DataTaxonomyServiceAsyncClient])
@mock.patch.object(DataTaxonomyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTaxonomyServiceClient))
@mock.patch.object(DataTaxonomyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTaxonomyServiceAsyncClient))
def test_data_taxonomy_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport, 'grpc'), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_data_taxonomy_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport, 'grpc', grpc_helpers), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_data_taxonomy_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_data_taxonomy_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dataplex_v1.services.data_taxonomy_service.transports.DataTaxonomyServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DataTaxonomyServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport, 'grpc', grpc_helpers), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_data_taxonomy_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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

@pytest.mark.parametrize('request_type', [gcd_data_taxonomy.CreateDataTaxonomyRequest, dict])
def test_create_data_taxonomy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_data_taxonomy.CreateDataTaxonomyRequest()
    assert isinstance(response, future.Future)

def test_create_data_taxonomy_empty_call():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        client.create_data_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_data_taxonomy.CreateDataTaxonomyRequest()

@pytest.mark.asyncio
async def test_create_data_taxonomy_async(transport: str='grpc_asyncio', request_type=gcd_data_taxonomy.CreateDataTaxonomyRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_data_taxonomy.CreateDataTaxonomyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_data_taxonomy_async_from_dict():
    await test_create_data_taxonomy_async(request_type=dict)

def test_create_data_taxonomy_field_headers():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_data_taxonomy.CreateDataTaxonomyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_data_taxonomy_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_data_taxonomy.CreateDataTaxonomyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_data_taxonomy_flattened():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_data_taxonomy(parent='parent_value', data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), data_taxonomy_id='data_taxonomy_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].data_taxonomy
        mock_val = gcd_data_taxonomy.DataTaxonomy(name='name_value')
        assert arg == mock_val
        arg = args[0].data_taxonomy_id
        mock_val = 'data_taxonomy_id_value'
        assert arg == mock_val

def test_create_data_taxonomy_flattened_error():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_data_taxonomy(gcd_data_taxonomy.CreateDataTaxonomyRequest(), parent='parent_value', data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), data_taxonomy_id='data_taxonomy_id_value')

@pytest.mark.asyncio
async def test_create_data_taxonomy_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_data_taxonomy(parent='parent_value', data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), data_taxonomy_id='data_taxonomy_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].data_taxonomy
        mock_val = gcd_data_taxonomy.DataTaxonomy(name='name_value')
        assert arg == mock_val
        arg = args[0].data_taxonomy_id
        mock_val = 'data_taxonomy_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_data_taxonomy_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_data_taxonomy(gcd_data_taxonomy.CreateDataTaxonomyRequest(), parent='parent_value', data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), data_taxonomy_id='data_taxonomy_id_value')

@pytest.mark.parametrize('request_type', [gcd_data_taxonomy.UpdateDataTaxonomyRequest, dict])
def test_update_data_taxonomy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_data_taxonomy.UpdateDataTaxonomyRequest()
    assert isinstance(response, future.Future)

def test_update_data_taxonomy_empty_call():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        client.update_data_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_data_taxonomy.UpdateDataTaxonomyRequest()

@pytest.mark.asyncio
async def test_update_data_taxonomy_async(transport: str='grpc_asyncio', request_type=gcd_data_taxonomy.UpdateDataTaxonomyRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_data_taxonomy.UpdateDataTaxonomyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_data_taxonomy_async_from_dict():
    await test_update_data_taxonomy_async(request_type=dict)

def test_update_data_taxonomy_field_headers():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_data_taxonomy.UpdateDataTaxonomyRequest()
    request.data_taxonomy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'data_taxonomy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_data_taxonomy_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_data_taxonomy.UpdateDataTaxonomyRequest()
    request.data_taxonomy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'data_taxonomy.name=name_value') in kw['metadata']

def test_update_data_taxonomy_flattened():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_data_taxonomy(data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].data_taxonomy
        mock_val = gcd_data_taxonomy.DataTaxonomy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_data_taxonomy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_data_taxonomy(gcd_data_taxonomy.UpdateDataTaxonomyRequest(), data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_data_taxonomy_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_data_taxonomy(data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].data_taxonomy
        mock_val = gcd_data_taxonomy.DataTaxonomy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_data_taxonomy_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_data_taxonomy(gcd_data_taxonomy.UpdateDataTaxonomyRequest(), data_taxonomy=gcd_data_taxonomy.DataTaxonomy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [data_taxonomy.DeleteDataTaxonomyRequest, dict])
def test_delete_data_taxonomy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataTaxonomyRequest()
    assert isinstance(response, future.Future)

def test_delete_data_taxonomy_empty_call():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        client.delete_data_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataTaxonomyRequest()

@pytest.mark.asyncio
async def test_delete_data_taxonomy_async(transport: str='grpc_asyncio', request_type=data_taxonomy.DeleteDataTaxonomyRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataTaxonomyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_data_taxonomy_async_from_dict():
    await test_delete_data_taxonomy_async(request_type=dict)

def test_delete_data_taxonomy_field_headers():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.DeleteDataTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_data_taxonomy_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.DeleteDataTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_data_taxonomy_flattened():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_data_taxonomy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_data_taxonomy_flattened_error():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_data_taxonomy(data_taxonomy.DeleteDataTaxonomyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_data_taxonomy_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_data_taxonomy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_data_taxonomy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_data_taxonomy_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_data_taxonomy(data_taxonomy.DeleteDataTaxonomyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.ListDataTaxonomiesRequest, dict])
def test_list_data_taxonomies(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.return_value = data_taxonomy.ListDataTaxonomiesResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response = client.list_data_taxonomies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataTaxonomiesRequest()
    assert isinstance(response, pagers.ListDataTaxonomiesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_data_taxonomies_empty_call():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        client.list_data_taxonomies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataTaxonomiesRequest()

@pytest.mark.asyncio
async def test_list_data_taxonomies_async(transport: str='grpc_asyncio', request_type=data_taxonomy.ListDataTaxonomiesRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataTaxonomiesResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value']))
        response = await client.list_data_taxonomies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataTaxonomiesRequest()
    assert isinstance(response, pagers.ListDataTaxonomiesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

@pytest.mark.asyncio
async def test_list_data_taxonomies_async_from_dict():
    await test_list_data_taxonomies_async(request_type=dict)

def test_list_data_taxonomies_field_headers():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.ListDataTaxonomiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.return_value = data_taxonomy.ListDataTaxonomiesResponse()
        client.list_data_taxonomies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_data_taxonomies_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.ListDataTaxonomiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataTaxonomiesResponse())
        await client.list_data_taxonomies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_data_taxonomies_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.return_value = data_taxonomy.ListDataTaxonomiesResponse()
        client.list_data_taxonomies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_data_taxonomies_flattened_error():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_data_taxonomies(data_taxonomy.ListDataTaxonomiesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_data_taxonomies_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.return_value = data_taxonomy.ListDataTaxonomiesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataTaxonomiesResponse())
        response = await client.list_data_taxonomies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_data_taxonomies_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_data_taxonomies(data_taxonomy.ListDataTaxonomiesRequest(), parent='parent_value')

def test_list_data_taxonomies_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.side_effect = (data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()], next_page_token='abc'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[], next_page_token='def'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy()], next_page_token='ghi'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_data_taxonomies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, data_taxonomy.DataTaxonomy) for i in results))

def test_list_data_taxonomies_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__') as call:
        call.side_effect = (data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()], next_page_token='abc'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[], next_page_token='def'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy()], next_page_token='ghi'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()]), RuntimeError)
        pages = list(client.list_data_taxonomies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_data_taxonomies_async_pager():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()], next_page_token='abc'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[], next_page_token='def'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy()], next_page_token='ghi'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()]), RuntimeError)
        async_pager = await client.list_data_taxonomies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, data_taxonomy.DataTaxonomy) for i in responses))

@pytest.mark.asyncio
async def test_list_data_taxonomies_async_pages():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_taxonomies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()], next_page_token='abc'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[], next_page_token='def'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy()], next_page_token='ghi'), data_taxonomy.ListDataTaxonomiesResponse(data_taxonomies=[data_taxonomy.DataTaxonomy(), data_taxonomy.DataTaxonomy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_data_taxonomies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [data_taxonomy.GetDataTaxonomyRequest, dict])
def test_get_data_taxonomy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        call.return_value = data_taxonomy.DataTaxonomy(name='name_value', uid='uid_value', description='description_value', display_name='display_name_value', attribute_count=1628, etag='etag_value', class_count=1182)
        response = client.get_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataTaxonomyRequest()
    assert isinstance(response, data_taxonomy.DataTaxonomy)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'
    assert response.attribute_count == 1628
    assert response.etag == 'etag_value'
    assert response.class_count == 1182

def test_get_data_taxonomy_empty_call():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        client.get_data_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataTaxonomyRequest()

@pytest.mark.asyncio
async def test_get_data_taxonomy_async(transport: str='grpc_asyncio', request_type=data_taxonomy.GetDataTaxonomyRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataTaxonomy(name='name_value', uid='uid_value', description='description_value', display_name='display_name_value', attribute_count=1628, etag='etag_value', class_count=1182))
        response = await client.get_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataTaxonomyRequest()
    assert isinstance(response, data_taxonomy.DataTaxonomy)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'
    assert response.attribute_count == 1628
    assert response.etag == 'etag_value'
    assert response.class_count == 1182

@pytest.mark.asyncio
async def test_get_data_taxonomy_async_from_dict():
    await test_get_data_taxonomy_async(request_type=dict)

def test_get_data_taxonomy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.GetDataTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        call.return_value = data_taxonomy.DataTaxonomy()
        client.get_data_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_data_taxonomy_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.GetDataTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataTaxonomy())
        await client.get_data_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_data_taxonomy_flattened():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        call.return_value = data_taxonomy.DataTaxonomy()
        client.get_data_taxonomy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_data_taxonomy_flattened_error():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_data_taxonomy(data_taxonomy.GetDataTaxonomyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_data_taxonomy_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_taxonomy), '__call__') as call:
        call.return_value = data_taxonomy.DataTaxonomy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataTaxonomy())
        response = await client.get_data_taxonomy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_data_taxonomy_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_data_taxonomy(data_taxonomy.GetDataTaxonomyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.CreateDataAttributeBindingRequest, dict])
def test_create_data_attribute_binding(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.CreateDataAttributeBindingRequest()
    assert isinstance(response, future.Future)

def test_create_data_attribute_binding_empty_call():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        client.create_data_attribute_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.CreateDataAttributeBindingRequest()

@pytest.mark.asyncio
async def test_create_data_attribute_binding_async(transport: str='grpc_asyncio', request_type=data_taxonomy.CreateDataAttributeBindingRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.CreateDataAttributeBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_data_attribute_binding_async_from_dict():
    await test_create_data_attribute_binding_async(request_type=dict)

def test_create_data_attribute_binding_field_headers():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.CreateDataAttributeBindingRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_data_attribute_binding_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.CreateDataAttributeBindingRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_data_attribute_binding_flattened():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_data_attribute_binding(parent='parent_value', data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), data_attribute_binding_id='data_attribute_binding_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].data_attribute_binding
        mock_val = data_taxonomy.DataAttributeBinding(name='name_value')
        assert arg == mock_val
        arg = args[0].data_attribute_binding_id
        mock_val = 'data_attribute_binding_id_value'
        assert arg == mock_val

def test_create_data_attribute_binding_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_data_attribute_binding(data_taxonomy.CreateDataAttributeBindingRequest(), parent='parent_value', data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), data_attribute_binding_id='data_attribute_binding_id_value')

@pytest.mark.asyncio
async def test_create_data_attribute_binding_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_data_attribute_binding(parent='parent_value', data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), data_attribute_binding_id='data_attribute_binding_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].data_attribute_binding
        mock_val = data_taxonomy.DataAttributeBinding(name='name_value')
        assert arg == mock_val
        arg = args[0].data_attribute_binding_id
        mock_val = 'data_attribute_binding_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_data_attribute_binding_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_data_attribute_binding(data_taxonomy.CreateDataAttributeBindingRequest(), parent='parent_value', data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), data_attribute_binding_id='data_attribute_binding_id_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.UpdateDataAttributeBindingRequest, dict])
def test_update_data_attribute_binding(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.UpdateDataAttributeBindingRequest()
    assert isinstance(response, future.Future)

def test_update_data_attribute_binding_empty_call():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        client.update_data_attribute_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.UpdateDataAttributeBindingRequest()

@pytest.mark.asyncio
async def test_update_data_attribute_binding_async(transport: str='grpc_asyncio', request_type=data_taxonomy.UpdateDataAttributeBindingRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.UpdateDataAttributeBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_data_attribute_binding_async_from_dict():
    await test_update_data_attribute_binding_async(request_type=dict)

def test_update_data_attribute_binding_field_headers():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.UpdateDataAttributeBindingRequest()
    request.data_attribute_binding.name = 'name_value'
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'data_attribute_binding.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_data_attribute_binding_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.UpdateDataAttributeBindingRequest()
    request.data_attribute_binding.name = 'name_value'
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'data_attribute_binding.name=name_value') in kw['metadata']

def test_update_data_attribute_binding_flattened():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_data_attribute_binding(data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].data_attribute_binding
        mock_val = data_taxonomy.DataAttributeBinding(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_data_attribute_binding_flattened_error():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_data_attribute_binding(data_taxonomy.UpdateDataAttributeBindingRequest(), data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_data_attribute_binding_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_data_attribute_binding(data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].data_attribute_binding
        mock_val = data_taxonomy.DataAttributeBinding(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_data_attribute_binding_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_data_attribute_binding(data_taxonomy.UpdateDataAttributeBindingRequest(), data_attribute_binding=data_taxonomy.DataAttributeBinding(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [data_taxonomy.DeleteDataAttributeBindingRequest, dict])
def test_delete_data_attribute_binding(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataAttributeBindingRequest()
    assert isinstance(response, future.Future)

def test_delete_data_attribute_binding_empty_call():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        client.delete_data_attribute_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataAttributeBindingRequest()

@pytest.mark.asyncio
async def test_delete_data_attribute_binding_async(transport: str='grpc_asyncio', request_type=data_taxonomy.DeleteDataAttributeBindingRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataAttributeBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_data_attribute_binding_async_from_dict():
    await test_delete_data_attribute_binding_async(request_type=dict)

def test_delete_data_attribute_binding_field_headers():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.DeleteDataAttributeBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_data_attribute_binding_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.DeleteDataAttributeBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_data_attribute_binding_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_data_attribute_binding(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_data_attribute_binding_flattened_error():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_data_attribute_binding(data_taxonomy.DeleteDataAttributeBindingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_data_attribute_binding_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_data_attribute_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_data_attribute_binding(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_data_attribute_binding_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_data_attribute_binding(data_taxonomy.DeleteDataAttributeBindingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.ListDataAttributeBindingsRequest, dict])
def test_list_data_attribute_bindings(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributeBindingsResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response = client.list_data_attribute_bindings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataAttributeBindingsRequest()
    assert isinstance(response, pagers.ListDataAttributeBindingsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_data_attribute_bindings_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        client.list_data_attribute_bindings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataAttributeBindingsRequest()

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_async(transport: str='grpc_asyncio', request_type=data_taxonomy.ListDataAttributeBindingsRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataAttributeBindingsResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value']))
        response = await client.list_data_attribute_bindings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataAttributeBindingsRequest()
    assert isinstance(response, pagers.ListDataAttributeBindingsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_async_from_dict():
    await test_list_data_attribute_bindings_async(request_type=dict)

def test_list_data_attribute_bindings_field_headers():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.ListDataAttributeBindingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributeBindingsResponse()
        client.list_data_attribute_bindings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.ListDataAttributeBindingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataAttributeBindingsResponse())
        await client.list_data_attribute_bindings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_data_attribute_bindings_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributeBindingsResponse()
        client.list_data_attribute_bindings(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_data_attribute_bindings_flattened_error():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_data_attribute_bindings(data_taxonomy.ListDataAttributeBindingsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributeBindingsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataAttributeBindingsResponse())
        response = await client.list_data_attribute_bindings(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_data_attribute_bindings(data_taxonomy.ListDataAttributeBindingsRequest(), parent='parent_value')

def test_list_data_attribute_bindings_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.side_effect = (data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()], next_page_token='abc'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[], next_page_token='def'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding()], next_page_token='ghi'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_data_attribute_bindings(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, data_taxonomy.DataAttributeBinding) for i in results))

def test_list_data_attribute_bindings_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__') as call:
        call.side_effect = (data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()], next_page_token='abc'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[], next_page_token='def'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding()], next_page_token='ghi'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()]), RuntimeError)
        pages = list(client.list_data_attribute_bindings(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_async_pager():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()], next_page_token='abc'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[], next_page_token='def'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding()], next_page_token='ghi'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()]), RuntimeError)
        async_pager = await client.list_data_attribute_bindings(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, data_taxonomy.DataAttributeBinding) for i in responses))

@pytest.mark.asyncio
async def test_list_data_attribute_bindings_async_pages():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_attribute_bindings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()], next_page_token='abc'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[], next_page_token='def'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding()], next_page_token='ghi'), data_taxonomy.ListDataAttributeBindingsResponse(data_attribute_bindings=[data_taxonomy.DataAttributeBinding(), data_taxonomy.DataAttributeBinding()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_data_attribute_bindings(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [data_taxonomy.GetDataAttributeBindingRequest, dict])
def test_get_data_attribute_binding(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        call.return_value = data_taxonomy.DataAttributeBinding(name='name_value', uid='uid_value', description='description_value', display_name='display_name_value', etag='etag_value', attributes=['attributes_value'], resource='resource_value')
        response = client.get_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataAttributeBindingRequest()
    assert isinstance(response, data_taxonomy.DataAttributeBinding)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'
    assert response.attributes == ['attributes_value']

def test_get_data_attribute_binding_empty_call():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        client.get_data_attribute_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataAttributeBindingRequest()

@pytest.mark.asyncio
async def test_get_data_attribute_binding_async(transport: str='grpc_asyncio', request_type=data_taxonomy.GetDataAttributeBindingRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataAttributeBinding(name='name_value', uid='uid_value', description='description_value', display_name='display_name_value', etag='etag_value', attributes=['attributes_value']))
        response = await client.get_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataAttributeBindingRequest()
    assert isinstance(response, data_taxonomy.DataAttributeBinding)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'
    assert response.attributes == ['attributes_value']

@pytest.mark.asyncio
async def test_get_data_attribute_binding_async_from_dict():
    await test_get_data_attribute_binding_async(request_type=dict)

def test_get_data_attribute_binding_field_headers():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.GetDataAttributeBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        call.return_value = data_taxonomy.DataAttributeBinding()
        client.get_data_attribute_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_data_attribute_binding_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.GetDataAttributeBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataAttributeBinding())
        await client.get_data_attribute_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_data_attribute_binding_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        call.return_value = data_taxonomy.DataAttributeBinding()
        client.get_data_attribute_binding(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_data_attribute_binding_flattened_error():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_data_attribute_binding(data_taxonomy.GetDataAttributeBindingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_data_attribute_binding_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_attribute_binding), '__call__') as call:
        call.return_value = data_taxonomy.DataAttributeBinding()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataAttributeBinding())
        response = await client.get_data_attribute_binding(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_data_attribute_binding_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_data_attribute_binding(data_taxonomy.GetDataAttributeBindingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.CreateDataAttributeRequest, dict])
def test_create_data_attribute(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.CreateDataAttributeRequest()
    assert isinstance(response, future.Future)

def test_create_data_attribute_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        client.create_data_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.CreateDataAttributeRequest()

@pytest.mark.asyncio
async def test_create_data_attribute_async(transport: str='grpc_asyncio', request_type=data_taxonomy.CreateDataAttributeRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.CreateDataAttributeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_data_attribute_async_from_dict():
    await test_create_data_attribute_async(request_type=dict)

def test_create_data_attribute_field_headers():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.CreateDataAttributeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_data_attribute_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.CreateDataAttributeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_data_attribute_flattened():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_data_attribute(parent='parent_value', data_attribute=data_taxonomy.DataAttribute(name='name_value'), data_attribute_id='data_attribute_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].data_attribute
        mock_val = data_taxonomy.DataAttribute(name='name_value')
        assert arg == mock_val
        arg = args[0].data_attribute_id
        mock_val = 'data_attribute_id_value'
        assert arg == mock_val

def test_create_data_attribute_flattened_error():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_data_attribute(data_taxonomy.CreateDataAttributeRequest(), parent='parent_value', data_attribute=data_taxonomy.DataAttribute(name='name_value'), data_attribute_id='data_attribute_id_value')

@pytest.mark.asyncio
async def test_create_data_attribute_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_data_attribute(parent='parent_value', data_attribute=data_taxonomy.DataAttribute(name='name_value'), data_attribute_id='data_attribute_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].data_attribute
        mock_val = data_taxonomy.DataAttribute(name='name_value')
        assert arg == mock_val
        arg = args[0].data_attribute_id
        mock_val = 'data_attribute_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_data_attribute_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_data_attribute(data_taxonomy.CreateDataAttributeRequest(), parent='parent_value', data_attribute=data_taxonomy.DataAttribute(name='name_value'), data_attribute_id='data_attribute_id_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.UpdateDataAttributeRequest, dict])
def test_update_data_attribute(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.UpdateDataAttributeRequest()
    assert isinstance(response, future.Future)

def test_update_data_attribute_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        client.update_data_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.UpdateDataAttributeRequest()

@pytest.mark.asyncio
async def test_update_data_attribute_async(transport: str='grpc_asyncio', request_type=data_taxonomy.UpdateDataAttributeRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.UpdateDataAttributeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_data_attribute_async_from_dict():
    await test_update_data_attribute_async(request_type=dict)

def test_update_data_attribute_field_headers():
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.UpdateDataAttributeRequest()
    request.data_attribute.name = 'name_value'
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'data_attribute.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_data_attribute_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.UpdateDataAttributeRequest()
    request.data_attribute.name = 'name_value'
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'data_attribute.name=name_value') in kw['metadata']

def test_update_data_attribute_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_data_attribute(data_attribute=data_taxonomy.DataAttribute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].data_attribute
        mock_val = data_taxonomy.DataAttribute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_data_attribute_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_data_attribute(data_taxonomy.UpdateDataAttributeRequest(), data_attribute=data_taxonomy.DataAttribute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_data_attribute_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_data_attribute(data_attribute=data_taxonomy.DataAttribute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].data_attribute
        mock_val = data_taxonomy.DataAttribute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_data_attribute_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_data_attribute(data_taxonomy.UpdateDataAttributeRequest(), data_attribute=data_taxonomy.DataAttribute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [data_taxonomy.DeleteDataAttributeRequest, dict])
def test_delete_data_attribute(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataAttributeRequest()
    assert isinstance(response, future.Future)

def test_delete_data_attribute_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        client.delete_data_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataAttributeRequest()

@pytest.mark.asyncio
async def test_delete_data_attribute_async(transport: str='grpc_asyncio', request_type=data_taxonomy.DeleteDataAttributeRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.DeleteDataAttributeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_data_attribute_async_from_dict():
    await test_delete_data_attribute_async(request_type=dict)

def test_delete_data_attribute_field_headers():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.DeleteDataAttributeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_data_attribute_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.DeleteDataAttributeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_data_attribute_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_data_attribute(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_data_attribute_flattened_error():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_data_attribute(data_taxonomy.DeleteDataAttributeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_data_attribute_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_data_attribute), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_data_attribute(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_data_attribute_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_data_attribute(data_taxonomy.DeleteDataAttributeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [data_taxonomy.ListDataAttributesRequest, dict])
def test_list_data_attributes(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributesResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response = client.list_data_attributes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataAttributesRequest()
    assert isinstance(response, pagers.ListDataAttributesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_data_attributes_empty_call():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        client.list_data_attributes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataAttributesRequest()

@pytest.mark.asyncio
async def test_list_data_attributes_async(transport: str='grpc_asyncio', request_type=data_taxonomy.ListDataAttributesRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataAttributesResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value']))
        response = await client.list_data_attributes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.ListDataAttributesRequest()
    assert isinstance(response, pagers.ListDataAttributesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

@pytest.mark.asyncio
async def test_list_data_attributes_async_from_dict():
    await test_list_data_attributes_async(request_type=dict)

def test_list_data_attributes_field_headers():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.ListDataAttributesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributesResponse()
        client.list_data_attributes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_data_attributes_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.ListDataAttributesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataAttributesResponse())
        await client.list_data_attributes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_data_attributes_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributesResponse()
        client.list_data_attributes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_data_attributes_flattened_error():
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_data_attributes(data_taxonomy.ListDataAttributesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_data_attributes_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.return_value = data_taxonomy.ListDataAttributesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.ListDataAttributesResponse())
        response = await client.list_data_attributes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_data_attributes_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_data_attributes(data_taxonomy.ListDataAttributesRequest(), parent='parent_value')

def test_list_data_attributes_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.side_effect = (data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()], next_page_token='abc'), data_taxonomy.ListDataAttributesResponse(data_attributes=[], next_page_token='def'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute()], next_page_token='ghi'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_data_attributes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, data_taxonomy.DataAttribute) for i in results))

def test_list_data_attributes_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__') as call:
        call.side_effect = (data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()], next_page_token='abc'), data_taxonomy.ListDataAttributesResponse(data_attributes=[], next_page_token='def'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute()], next_page_token='ghi'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()]), RuntimeError)
        pages = list(client.list_data_attributes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_data_attributes_async_pager():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()], next_page_token='abc'), data_taxonomy.ListDataAttributesResponse(data_attributes=[], next_page_token='def'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute()], next_page_token='ghi'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()]), RuntimeError)
        async_pager = await client.list_data_attributes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, data_taxonomy.DataAttribute) for i in responses))

@pytest.mark.asyncio
async def test_list_data_attributes_async_pages():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_attributes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()], next_page_token='abc'), data_taxonomy.ListDataAttributesResponse(data_attributes=[], next_page_token='def'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute()], next_page_token='ghi'), data_taxonomy.ListDataAttributesResponse(data_attributes=[data_taxonomy.DataAttribute(), data_taxonomy.DataAttribute()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_data_attributes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [data_taxonomy.GetDataAttributeRequest, dict])
def test_get_data_attribute(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        call.return_value = data_taxonomy.DataAttribute(name='name_value', uid='uid_value', description='description_value', display_name='display_name_value', parent_id='parent_id_value', attribute_count=1628, etag='etag_value')
        response = client.get_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataAttributeRequest()
    assert isinstance(response, data_taxonomy.DataAttribute)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'
    assert response.parent_id == 'parent_id_value'
    assert response.attribute_count == 1628
    assert response.etag == 'etag_value'

def test_get_data_attribute_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        client.get_data_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataAttributeRequest()

@pytest.mark.asyncio
async def test_get_data_attribute_async(transport: str='grpc_asyncio', request_type=data_taxonomy.GetDataAttributeRequest):
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataAttribute(name='name_value', uid='uid_value', description='description_value', display_name='display_name_value', parent_id='parent_id_value', attribute_count=1628, etag='etag_value'))
        response = await client.get_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == data_taxonomy.GetDataAttributeRequest()
    assert isinstance(response, data_taxonomy.DataAttribute)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.display_name == 'display_name_value'
    assert response.parent_id == 'parent_id_value'
    assert response.attribute_count == 1628
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_data_attribute_async_from_dict():
    await test_get_data_attribute_async(request_type=dict)

def test_get_data_attribute_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.GetDataAttributeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        call.return_value = data_taxonomy.DataAttribute()
        client.get_data_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_data_attribute_field_headers_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = data_taxonomy.GetDataAttributeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataAttribute())
        await client.get_data_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_data_attribute_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        call.return_value = data_taxonomy.DataAttribute()
        client.get_data_attribute(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_data_attribute_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_data_attribute(data_taxonomy.GetDataAttributeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_data_attribute_flattened_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_attribute), '__call__') as call:
        call.return_value = data_taxonomy.DataAttribute()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(data_taxonomy.DataAttribute())
        response = await client.get_data_attribute(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_data_attribute_flattened_error_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_data_attribute(data_taxonomy.GetDataAttributeRequest(), name='name_value')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.DataTaxonomyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DataTaxonomyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DataTaxonomyServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DataTaxonomyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DataTaxonomyServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DataTaxonomyServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DataTaxonomyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DataTaxonomyServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTaxonomyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DataTaxonomyServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.DataTaxonomyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DataTaxonomyServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DataTaxonomyServiceGrpcTransport, transports.DataTaxonomyServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        return 10
    transport = DataTaxonomyServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DataTaxonomyServiceGrpcTransport)

def test_data_taxonomy_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DataTaxonomyServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_data_taxonomy_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dataplex_v1.services.data_taxonomy_service.transports.DataTaxonomyServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DataTaxonomyServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_data_taxonomy', 'update_data_taxonomy', 'delete_data_taxonomy', 'list_data_taxonomies', 'get_data_taxonomy', 'create_data_attribute_binding', 'update_data_attribute_binding', 'delete_data_attribute_binding', 'list_data_attribute_bindings', 'get_data_attribute_binding', 'create_data_attribute', 'update_data_attribute', 'delete_data_attribute', 'list_data_attributes', 'get_data_attribute', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_data_taxonomy_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataplex_v1.services.data_taxonomy_service.transports.DataTaxonomyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DataTaxonomyServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_data_taxonomy_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataplex_v1.services.data_taxonomy_service.transports.DataTaxonomyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DataTaxonomyServiceTransport()
        adc.assert_called_once()

def test_data_taxonomy_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DataTaxonomyServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DataTaxonomyServiceGrpcTransport, transports.DataTaxonomyServiceGrpcAsyncIOTransport])
def test_data_taxonomy_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DataTaxonomyServiceGrpcTransport, transports.DataTaxonomyServiceGrpcAsyncIOTransport])
def test_data_taxonomy_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DataTaxonomyServiceGrpcTransport, grpc_helpers), (transports.DataTaxonomyServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_data_taxonomy_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataplex.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dataplex.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DataTaxonomyServiceGrpcTransport, transports.DataTaxonomyServiceGrpcAsyncIOTransport])
def test_data_taxonomy_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_data_taxonomy_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataplex.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'dataplex.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_data_taxonomy_service_host_with_port(transport_name):
    if False:
        return 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataplex.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'dataplex.googleapis.com:8000'

def test_data_taxonomy_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DataTaxonomyServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_data_taxonomy_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DataTaxonomyServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DataTaxonomyServiceGrpcTransport, transports.DataTaxonomyServiceGrpcAsyncIOTransport])
def test_data_taxonomy_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.DataTaxonomyServiceGrpcTransport, transports.DataTaxonomyServiceGrpcAsyncIOTransport])
def test_data_taxonomy_service_transport_channel_mtls_with_adc(transport_class):
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

def test_data_taxonomy_service_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_data_taxonomy_service_grpc_lro_async_client():
    if False:
        return 10
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_data_attribute_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    dataTaxonomy = 'whelk'
    data_attribute_id = 'octopus'
    expected = 'projects/{project}/locations/{location}/dataTaxonomies/{dataTaxonomy}/attributes/{data_attribute_id}'.format(project=project, location=location, dataTaxonomy=dataTaxonomy, data_attribute_id=data_attribute_id)
    actual = DataTaxonomyServiceClient.data_attribute_path(project, location, dataTaxonomy, data_attribute_id)
    assert expected == actual

def test_parse_data_attribute_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'dataTaxonomy': 'cuttlefish', 'data_attribute_id': 'mussel'}
    path = DataTaxonomyServiceClient.data_attribute_path(**expected)
    actual = DataTaxonomyServiceClient.parse_data_attribute_path(path)
    assert expected == actual

def test_data_attribute_binding_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    data_attribute_binding_id = 'scallop'
    expected = 'projects/{project}/locations/{location}/dataAttributeBindings/{data_attribute_binding_id}'.format(project=project, location=location, data_attribute_binding_id=data_attribute_binding_id)
    actual = DataTaxonomyServiceClient.data_attribute_binding_path(project, location, data_attribute_binding_id)
    assert expected == actual

def test_parse_data_attribute_binding_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone', 'location': 'squid', 'data_attribute_binding_id': 'clam'}
    path = DataTaxonomyServiceClient.data_attribute_binding_path(**expected)
    actual = DataTaxonomyServiceClient.parse_data_attribute_binding_path(path)
    assert expected == actual

def test_data_taxonomy_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    data_taxonomy_id = 'oyster'
    expected = 'projects/{project}/locations/{location}/dataTaxonomies/{data_taxonomy_id}'.format(project=project, location=location, data_taxonomy_id=data_taxonomy_id)
    actual = DataTaxonomyServiceClient.data_taxonomy_path(project, location, data_taxonomy_id)
    assert expected == actual

def test_parse_data_taxonomy_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'data_taxonomy_id': 'mussel'}
    path = DataTaxonomyServiceClient.data_taxonomy_path(**expected)
    actual = DataTaxonomyServiceClient.parse_data_taxonomy_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DataTaxonomyServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'nautilus'}
    path = DataTaxonomyServiceClient.common_billing_account_path(**expected)
    actual = DataTaxonomyServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DataTaxonomyServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'abalone'}
    path = DataTaxonomyServiceClient.common_folder_path(**expected)
    actual = DataTaxonomyServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DataTaxonomyServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'clam'}
    path = DataTaxonomyServiceClient.common_organization_path(**expected)
    actual = DataTaxonomyServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = DataTaxonomyServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus'}
    path = DataTaxonomyServiceClient.common_project_path(**expected)
    actual = DataTaxonomyServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DataTaxonomyServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = DataTaxonomyServiceClient.common_location_path(**expected)
    actual = DataTaxonomyServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DataTaxonomyServiceTransport, '_prep_wrapped_messages') as prep:
        client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DataTaxonomyServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DataTaxonomyServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = DataTaxonomyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['grpc']
    for transport in transports:
        client = DataTaxonomyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DataTaxonomyServiceClient, transports.DataTaxonomyServiceGrpcTransport), (DataTaxonomyServiceAsyncClient, transports.DataTaxonomyServiceGrpcAsyncIOTransport)])
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