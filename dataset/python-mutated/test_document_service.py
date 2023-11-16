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
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.documentai_v1beta3.services.document_service import DocumentServiceAsyncClient, DocumentServiceClient, pagers, transports
from google.cloud.documentai_v1beta3.types import document, document_io, document_schema, document_service
from google.cloud.documentai_v1beta3.types import dataset
from google.cloud.documentai_v1beta3.types import dataset as gcd_dataset

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert DocumentServiceClient._get_default_mtls_endpoint(None) is None
    assert DocumentServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DocumentServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DocumentServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DocumentServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DocumentServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DocumentServiceClient, 'grpc'), (DocumentServiceAsyncClient, 'grpc_asyncio'), (DocumentServiceClient, 'rest')])
def test_document_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('documentai.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DocumentServiceGrpcTransport, 'grpc'), (transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.DocumentServiceRestTransport, 'rest')])
def test_document_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DocumentServiceClient, 'grpc'), (DocumentServiceAsyncClient, 'grpc_asyncio'), (DocumentServiceClient, 'rest')])
def test_document_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('documentai.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com')

def test_document_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = DocumentServiceClient.get_transport_class()
    available_transports = [transports.DocumentServiceGrpcTransport, transports.DocumentServiceRestTransport]
    assert transport in available_transports
    transport = DocumentServiceClient.get_transport_class('grpc')
    assert transport == transports.DocumentServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DocumentServiceClient, transports.DocumentServiceGrpcTransport, 'grpc'), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DocumentServiceClient, transports.DocumentServiceRestTransport, 'rest')])
@mock.patch.object(DocumentServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentServiceClient))
@mock.patch.object(DocumentServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentServiceAsyncClient))
def test_document_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(DocumentServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DocumentServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DocumentServiceClient, transports.DocumentServiceGrpcTransport, 'grpc', 'true'), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DocumentServiceClient, transports.DocumentServiceGrpcTransport, 'grpc', 'false'), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (DocumentServiceClient, transports.DocumentServiceRestTransport, 'rest', 'true'), (DocumentServiceClient, transports.DocumentServiceRestTransport, 'rest', 'false')])
@mock.patch.object(DocumentServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentServiceClient))
@mock.patch.object(DocumentServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_document_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [DocumentServiceClient, DocumentServiceAsyncClient])
@mock.patch.object(DocumentServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentServiceClient))
@mock.patch.object(DocumentServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentServiceAsyncClient))
def test_document_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DocumentServiceClient, transports.DocumentServiceGrpcTransport, 'grpc'), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DocumentServiceClient, transports.DocumentServiceRestTransport, 'rest')])
def test_document_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DocumentServiceClient, transports.DocumentServiceGrpcTransport, 'grpc', grpc_helpers), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (DocumentServiceClient, transports.DocumentServiceRestTransport, 'rest', None)])
def test_document_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_document_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.documentai_v1beta3.services.document_service.transports.DocumentServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DocumentServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DocumentServiceClient, transports.DocumentServiceGrpcTransport, 'grpc', grpc_helpers), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_document_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('documentai.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='documentai.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [document_service.UpdateDatasetRequest, dict])
def test_update_dataset(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.UpdateDatasetRequest()
    assert isinstance(response, future.Future)

def test_update_dataset_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        client.update_dataset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.UpdateDatasetRequest()

@pytest.mark.asyncio
async def test_update_dataset_async(transport: str='grpc_asyncio', request_type=document_service.UpdateDatasetRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.UpdateDatasetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_dataset_async_from_dict():
    await test_update_dataset_async(request_type=dict)

def test_update_dataset_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.UpdateDatasetRequest()
    request.dataset.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_dataset_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.UpdateDatasetRequest()
    request.dataset.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset.name=name_value') in kw['metadata']

def test_update_dataset_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_dataset(dataset=gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value')))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_dataset_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_dataset(document_service.UpdateDatasetRequest(), dataset=gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_dataset_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dataset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_dataset(dataset=gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value')))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_dataset_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_dataset(document_service.UpdateDatasetRequest(), dataset=gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [document_service.ImportDocumentsRequest, dict])
def test_import_documents(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.import_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.ImportDocumentsRequest()
    assert isinstance(response, future.Future)

def test_import_documents_empty_call():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        client.import_documents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.ImportDocumentsRequest()

@pytest.mark.asyncio
async def test_import_documents_async(transport: str='grpc_asyncio', request_type=document_service.ImportDocumentsRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.ImportDocumentsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_import_documents_async_from_dict():
    await test_import_documents_async(request_type=dict)

def test_import_documents_field_headers():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.ImportDocumentsRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_documents_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.ImportDocumentsRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.import_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

def test_import_documents_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_documents(dataset='dataset_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

def test_import_documents_flattened_error():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.import_documents(document_service.ImportDocumentsRequest(), dataset='dataset_value')

@pytest.mark.asyncio
async def test_import_documents_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_documents(dataset='dataset_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_import_documents_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.import_documents(document_service.ImportDocumentsRequest(), dataset='dataset_value')

@pytest.mark.parametrize('request_type', [document_service.GetDocumentRequest, dict])
def test_get_document(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        call.return_value = document_service.GetDocumentResponse()
        response = client.get_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.GetDocumentRequest()
    assert isinstance(response, document_service.GetDocumentResponse)

def test_get_document_empty_call():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        client.get_document()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.GetDocumentRequest()

@pytest.mark.asyncio
async def test_get_document_async(transport: str='grpc_asyncio', request_type=document_service.GetDocumentRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_service.GetDocumentResponse())
        response = await client.get_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.GetDocumentRequest()
    assert isinstance(response, document_service.GetDocumentResponse)

@pytest.mark.asyncio
async def test_get_document_async_from_dict():
    await test_get_document_async(request_type=dict)

def test_get_document_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.GetDocumentRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        call.return_value = document_service.GetDocumentResponse()
        client.get_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_document_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.GetDocumentRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_service.GetDocumentResponse())
        await client.get_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

def test_get_document_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        call.return_value = document_service.GetDocumentResponse()
        client.get_document(dataset='dataset_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

def test_get_document_flattened_error():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_document(document_service.GetDocumentRequest(), dataset='dataset_value')

@pytest.mark.asyncio
async def test_get_document_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_document), '__call__') as call:
        call.return_value = document_service.GetDocumentResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_service.GetDocumentResponse())
        response = await client.get_document(dataset='dataset_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_document_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_document(document_service.GetDocumentRequest(), dataset='dataset_value')

@pytest.mark.parametrize('request_type', [document_service.ListDocumentsRequest, dict])
def test_list_documents(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.return_value = document_service.ListDocumentsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.ListDocumentsRequest()
    assert isinstance(response, pagers.ListDocumentsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_documents_empty_call():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        client.list_documents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.ListDocumentsRequest()

@pytest.mark.asyncio
async def test_list_documents_async(transport: str='grpc_asyncio', request_type=document_service.ListDocumentsRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_service.ListDocumentsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.ListDocumentsRequest()
    assert isinstance(response, pagers.ListDocumentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_documents_async_from_dict():
    await test_list_documents_async(request_type=dict)

def test_list_documents_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.ListDocumentsRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.return_value = document_service.ListDocumentsResponse()
        client.list_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_documents_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.ListDocumentsRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_service.ListDocumentsResponse())
        await client.list_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

def test_list_documents_flattened():
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.return_value = document_service.ListDocumentsResponse()
        client.list_documents(dataset='dataset_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

def test_list_documents_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_documents(document_service.ListDocumentsRequest(), dataset='dataset_value')

@pytest.mark.asyncio
async def test_list_documents_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.return_value = document_service.ListDocumentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_service.ListDocumentsResponse())
        response = await client.list_documents(dataset='dataset_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_documents_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_documents(document_service.ListDocumentsRequest(), dataset='dataset_value')

def test_list_documents_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.side_effect = (document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata(), document_service.DocumentMetadata()], next_page_token='abc'), document_service.ListDocumentsResponse(document_metadata=[], next_page_token='def'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata()], next_page_token='ghi'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('dataset', ''),)),)
        pager = client.list_documents(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, document_service.DocumentMetadata) for i in results))

def test_list_documents_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_documents), '__call__') as call:
        call.side_effect = (document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata(), document_service.DocumentMetadata()], next_page_token='abc'), document_service.ListDocumentsResponse(document_metadata=[], next_page_token='def'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata()], next_page_token='ghi'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata()]), RuntimeError)
        pages = list(client.list_documents(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_documents_async_pager():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_documents), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata(), document_service.DocumentMetadata()], next_page_token='abc'), document_service.ListDocumentsResponse(document_metadata=[], next_page_token='def'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata()], next_page_token='ghi'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata()]), RuntimeError)
        async_pager = await client.list_documents(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, document_service.DocumentMetadata) for i in responses))

@pytest.mark.asyncio
async def test_list_documents_async_pages():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_documents), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata(), document_service.DocumentMetadata()], next_page_token='abc'), document_service.ListDocumentsResponse(document_metadata=[], next_page_token='def'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata()], next_page_token='ghi'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_documents(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_service.BatchDeleteDocumentsRequest, dict])
def test_batch_delete_documents(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_delete_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.BatchDeleteDocumentsRequest()
    assert isinstance(response, future.Future)

def test_batch_delete_documents_empty_call():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        client.batch_delete_documents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.BatchDeleteDocumentsRequest()

@pytest.mark.asyncio
async def test_batch_delete_documents_async(transport: str='grpc_asyncio', request_type=document_service.BatchDeleteDocumentsRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.BatchDeleteDocumentsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_delete_documents_async_from_dict():
    await test_batch_delete_documents_async(request_type=dict)

def test_batch_delete_documents_field_headers():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.BatchDeleteDocumentsRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_documents_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.BatchDeleteDocumentsRequest()
    request.dataset = 'dataset_value'
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_delete_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset=dataset_value') in kw['metadata']

def test_batch_delete_documents_flattened():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_documents(dataset='dataset_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

def test_batch_delete_documents_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_delete_documents(document_service.BatchDeleteDocumentsRequest(), dataset='dataset_value')

@pytest.mark.asyncio
async def test_batch_delete_documents_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_documents(dataset='dataset_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = 'dataset_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_delete_documents_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_delete_documents(document_service.BatchDeleteDocumentsRequest(), dataset='dataset_value')

@pytest.mark.parametrize('request_type', [document_service.GetDatasetSchemaRequest, dict])
def test_get_dataset_schema(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema(name='name_value')
        response = client.get_dataset_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.GetDatasetSchemaRequest()
    assert isinstance(response, dataset.DatasetSchema)
    assert response.name == 'name_value'

def test_get_dataset_schema_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        client.get_dataset_schema()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.GetDatasetSchemaRequest()

@pytest.mark.asyncio
async def test_get_dataset_schema_async(transport: str='grpc_asyncio', request_type=document_service.GetDatasetSchemaRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.DatasetSchema(name='name_value'))
        response = await client.get_dataset_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.GetDatasetSchemaRequest()
    assert isinstance(response, dataset.DatasetSchema)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_dataset_schema_async_from_dict():
    await test_get_dataset_schema_async(request_type=dict)

def test_get_dataset_schema_field_headers():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.GetDatasetSchemaRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema()
        client.get_dataset_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_dataset_schema_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.GetDatasetSchemaRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.DatasetSchema())
        await client.get_dataset_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_dataset_schema_flattened():
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema()
        client.get_dataset_schema(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_dataset_schema_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_dataset_schema(document_service.GetDatasetSchemaRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_dataset_schema_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.DatasetSchema())
        response = await client.get_dataset_schema(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_dataset_schema_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_dataset_schema(document_service.GetDatasetSchemaRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_service.UpdateDatasetSchemaRequest, dict])
def test_update_dataset_schema(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema(name='name_value')
        response = client.update_dataset_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.UpdateDatasetSchemaRequest()
    assert isinstance(response, dataset.DatasetSchema)
    assert response.name == 'name_value'

def test_update_dataset_schema_empty_call():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        client.update_dataset_schema()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.UpdateDatasetSchemaRequest()

@pytest.mark.asyncio
async def test_update_dataset_schema_async(transport: str='grpc_asyncio', request_type=document_service.UpdateDatasetSchemaRequest):
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.DatasetSchema(name='name_value'))
        response = await client.update_dataset_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_service.UpdateDatasetSchemaRequest()
    assert isinstance(response, dataset.DatasetSchema)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_dataset_schema_async_from_dict():
    await test_update_dataset_schema_async(request_type=dict)

def test_update_dataset_schema_field_headers():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.UpdateDatasetSchemaRequest()
    request.dataset_schema.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema()
        client.update_dataset_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset_schema.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_dataset_schema_field_headers_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_service.UpdateDatasetSchemaRequest()
    request.dataset_schema.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.DatasetSchema())
        await client.update_dataset_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset_schema.name=name_value') in kw['metadata']

def test_update_dataset_schema_flattened():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema()
        client.update_dataset_schema(dataset_schema=dataset.DatasetSchema(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset_schema
        mock_val = dataset.DatasetSchema(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_dataset_schema_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_dataset_schema(document_service.UpdateDatasetSchemaRequest(), dataset_schema=dataset.DatasetSchema(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_dataset_schema_flattened_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dataset_schema), '__call__') as call:
        call.return_value = dataset.DatasetSchema()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.DatasetSchema())
        response = await client.update_dataset_schema(dataset_schema=dataset.DatasetSchema(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset_schema
        mock_val = dataset.DatasetSchema(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_dataset_schema_flattened_error_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_dataset_schema(document_service.UpdateDatasetSchemaRequest(), dataset_schema=dataset.DatasetSchema(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [document_service.UpdateDatasetRequest, dict])
def test_update_dataset_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset': {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset'}}
    request_init['dataset'] = {'gcs_managed_config': {'gcs_prefix': {'gcs_uri_prefix': 'gcs_uri_prefix_value'}}, 'document_warehouse_config': {'collection': 'collection_value', 'schema': 'schema_value'}, 'unmanaged_dataset_config': {}, 'spanner_indexing_config': {}, 'name': 'projects/sample1/locations/sample2/processors/sample3/dataset', 'state': 1}
    test_field = document_service.UpdateDatasetRequest.meta.fields['dataset']

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
    for (field, value) in request_init['dataset'].items():
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
                for i in range(0, len(request_init['dataset'][field])):
                    del request_init['dataset'][field][i][subfield]
            else:
                del request_init['dataset'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_dataset(request)
    assert response.operation.name == 'operations/spam'

def test_update_dataset_rest_required_fields(request_type=document_service.UpdateDatasetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dataset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dataset._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_dataset(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_dataset_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_dataset._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('dataset',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_dataset_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_update_dataset') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_update_dataset') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.UpdateDatasetRequest.pb(document_service.UpdateDatasetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_service.UpdateDatasetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_dataset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_dataset_rest_bad_request(transport: str='rest', request_type=document_service.UpdateDatasetRequest):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset': {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_dataset(request)

def test_update_dataset_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'dataset': {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset'}}
        mock_args = dict(dataset=gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_dataset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{dataset.name=projects/*/locations/*/processors/*/dataset}' % client.transport._host, args[1])

def test_update_dataset_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_dataset(document_service.UpdateDatasetRequest(), dataset=gcd_dataset.Dataset(gcs_managed_config=gcd_dataset.Dataset.GCSManagedConfig(gcs_prefix=document_io.GcsPrefix(gcs_uri_prefix='gcs_uri_prefix_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_dataset_rest_error():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_service.ImportDocumentsRequest, dict])
def test_import_documents_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_documents(request)
    assert response.operation.name == 'operations/spam'

def test_import_documents_rest_required_fields(request_type=document_service.ImportDocumentsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request_init['dataset'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['dataset'] = 'dataset_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'dataset' in jsonified_request
    assert jsonified_request['dataset'] == 'dataset_value'
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.import_documents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_documents_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_documents._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('dataset', 'batchDocumentsImportConfigs'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_documents_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_import_documents') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_import_documents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.ImportDocumentsRequest.pb(document_service.ImportDocumentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_service.ImportDocumentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.import_documents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_documents_rest_bad_request(transport: str='rest', request_type=document_service.ImportDocumentsRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_documents(request)

def test_import_documents_rest_flattened():
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
        mock_args = dict(dataset='dataset_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.import_documents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{dataset=projects/*/locations/*/processors/*/dataset}:importDocuments' % client.transport._host, args[1])

def test_import_documents_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.import_documents(document_service.ImportDocumentsRequest(), dataset='dataset_value')

def test_import_documents_rest_error():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_service.GetDocumentRequest, dict])
def test_get_document_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_service.GetDocumentResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = document_service.GetDocumentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_document(request)
    assert isinstance(response, document_service.GetDocumentResponse)

def test_get_document_rest_required_fields(request_type=document_service.GetDocumentRequest):
    if False:
        return 10
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request_init['dataset'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['dataset'] = 'dataset_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_document._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('document_id', 'page_range', 'read_mask'))
    jsonified_request.update(unset_fields)
    assert 'dataset' in jsonified_request
    assert jsonified_request['dataset'] == 'dataset_value'
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_service.GetDocumentResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_service.GetDocumentResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_document(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_document_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_document._get_unset_required_fields({})
    assert set(unset_fields) == set(('documentId', 'pageRange', 'readMask')) & set(('dataset', 'documentId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_document_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_get_document') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_get_document') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.GetDocumentRequest.pb(document_service.GetDocumentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_service.GetDocumentResponse.to_json(document_service.GetDocumentResponse())
        request = document_service.GetDocumentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_service.GetDocumentResponse()
        client.get_document(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_document_rest_bad_request(transport: str='rest', request_type=document_service.GetDocumentRequest):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_document(request)

def test_get_document_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_service.GetDocumentResponse()
        sample_request = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
        mock_args = dict(dataset='dataset_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_service.GetDocumentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_document(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{dataset=projects/*/locations/*/processors/*/dataset}:getDocument' % client.transport._host, args[1])

def test_get_document_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_document(document_service.GetDocumentRequest(), dataset='dataset_value')

def test_get_document_rest_error():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_service.ListDocumentsRequest, dict])
def test_list_documents_rest(request_type):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_service.ListDocumentsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_service.ListDocumentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_documents(request)
    assert isinstance(response, pagers.ListDocumentsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_documents_rest_required_fields(request_type=document_service.ListDocumentsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request_init['dataset'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['dataset'] = 'dataset_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'dataset' in jsonified_request
    assert jsonified_request['dataset'] == 'dataset_value'
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_service.ListDocumentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_service.ListDocumentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_documents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_documents_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_documents._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('dataset',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_documents_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_list_documents') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_list_documents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.ListDocumentsRequest.pb(document_service.ListDocumentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_service.ListDocumentsResponse.to_json(document_service.ListDocumentsResponse())
        request = document_service.ListDocumentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_service.ListDocumentsResponse()
        client.list_documents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_documents_rest_bad_request(transport: str='rest', request_type=document_service.ListDocumentsRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_documents(request)

def test_list_documents_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_service.ListDocumentsResponse()
        sample_request = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
        mock_args = dict(dataset='dataset_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_service.ListDocumentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_documents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{dataset=projects/*/locations/*/processors/*/dataset}:listDocuments' % client.transport._host, args[1])

def test_list_documents_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_documents(document_service.ListDocumentsRequest(), dataset='dataset_value')

def test_list_documents_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata(), document_service.DocumentMetadata()], next_page_token='abc'), document_service.ListDocumentsResponse(document_metadata=[], next_page_token='def'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata()], next_page_token='ghi'), document_service.ListDocumentsResponse(document_metadata=[document_service.DocumentMetadata(), document_service.DocumentMetadata()]))
        response = response + response
        response = tuple((document_service.ListDocumentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
        pager = client.list_documents(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, document_service.DocumentMetadata) for i in results))
        pages = list(client.list_documents(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_service.BatchDeleteDocumentsRequest, dict])
def test_batch_delete_documents_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_documents(request)
    assert response.operation.name == 'operations/spam'

def test_batch_delete_documents_rest_required_fields(request_type=document_service.BatchDeleteDocumentsRequest):
    if False:
        return 10
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request_init['dataset'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['dataset'] = 'dataset_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'dataset' in jsonified_request
    assert jsonified_request['dataset'] == 'dataset_value'
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_delete_documents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_documents_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_documents._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('dataset', 'datasetDocuments'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_documents_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_batch_delete_documents') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_batch_delete_documents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.BatchDeleteDocumentsRequest.pb(document_service.BatchDeleteDocumentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_service.BatchDeleteDocumentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_delete_documents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_delete_documents_rest_bad_request(transport: str='rest', request_type=document_service.BatchDeleteDocumentsRequest):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_documents(request)

def test_batch_delete_documents_rest_flattened():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'dataset': 'projects/sample1/locations/sample2/processors/sample3/dataset'}
        mock_args = dict(dataset='dataset_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_delete_documents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{dataset=projects/*/locations/*/processors/*/dataset}:batchDeleteDocuments' % client.transport._host, args[1])

def test_batch_delete_documents_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_delete_documents(document_service.BatchDeleteDocumentsRequest(), dataset='dataset_value')

def test_batch_delete_documents_rest_error():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_service.GetDatasetSchemaRequest, dict])
def test_get_dataset_schema_rest(request_type):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dataset.DatasetSchema(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dataset.DatasetSchema.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_dataset_schema(request)
    assert isinstance(response, dataset.DatasetSchema)
    assert response.name == 'name_value'

def test_get_dataset_schema_rest_required_fields(request_type=document_service.GetDatasetSchemaRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dataset_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dataset_schema._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('visible_fields_only',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dataset.DatasetSchema()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dataset.DatasetSchema.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_dataset_schema(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_dataset_schema_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_dataset_schema._get_unset_required_fields({})
    assert set(unset_fields) == set(('visibleFieldsOnly',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_dataset_schema_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_get_dataset_schema') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_get_dataset_schema') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.GetDatasetSchemaRequest.pb(document_service.GetDatasetSchemaRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dataset.DatasetSchema.to_json(dataset.DatasetSchema())
        request = document_service.GetDatasetSchemaRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dataset.DatasetSchema()
        client.get_dataset_schema(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_dataset_schema_rest_bad_request(transport: str='rest', request_type=document_service.GetDatasetSchemaRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_dataset_schema(request)

def test_get_dataset_schema_rest_flattened():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dataset.DatasetSchema()
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dataset.DatasetSchema.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_dataset_schema(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{name=projects/*/locations/*/processors/*/dataset/datasetSchema}' % client.transport._host, args[1])

def test_get_dataset_schema_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_dataset_schema(document_service.GetDatasetSchemaRequest(), name='name_value')

def test_get_dataset_schema_rest_error():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_service.UpdateDatasetSchemaRequest, dict])
def test_update_dataset_schema_rest(request_type):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset_schema': {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema'}}
    request_init['dataset_schema'] = {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema', 'document_schema': {'display_name': 'display_name_value', 'description': 'description_value', 'entity_types': [{'enum_values': {'values': ['values_value1', 'values_value2']}, 'display_name': 'display_name_value', 'name': 'name_value', 'base_types': ['base_types_value1', 'base_types_value2'], 'properties': [{'name': 'name_value', 'value_type': 'value_type_value', 'occurrence_type': 1, 'property_metadata': {'inactive': True, 'field_extraction_metadata': {'summary_options': {'length': 1, 'format_': 1}}}}], 'entity_type_metadata': {'inactive': True}}], 'metadata': {'document_splitter': True, 'document_allow_multiple_labels': True, 'prefixed_naming_on_properties': True, 'skip_naming_validation': True}}}
    test_field = document_service.UpdateDatasetSchemaRequest.meta.fields['dataset_schema']

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
    for (field, value) in request_init['dataset_schema'].items():
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
                for i in range(0, len(request_init['dataset_schema'][field])):
                    del request_init['dataset_schema'][field][i][subfield]
            else:
                del request_init['dataset_schema'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dataset.DatasetSchema(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dataset.DatasetSchema.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_dataset_schema(request)
    assert isinstance(response, dataset.DatasetSchema)
    assert response.name == 'name_value'

def test_update_dataset_schema_rest_required_fields(request_type=document_service.UpdateDatasetSchemaRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DocumentServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dataset_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dataset_schema._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dataset.DatasetSchema()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dataset.DatasetSchema.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_dataset_schema(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_dataset_schema_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_dataset_schema._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('datasetSchema',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_dataset_schema_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DocumentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentServiceRestInterceptor())
    client = DocumentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentServiceRestInterceptor, 'post_update_dataset_schema') as post, mock.patch.object(transports.DocumentServiceRestInterceptor, 'pre_update_dataset_schema') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_service.UpdateDatasetSchemaRequest.pb(document_service.UpdateDatasetSchemaRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dataset.DatasetSchema.to_json(dataset.DatasetSchema())
        request = document_service.UpdateDatasetSchemaRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dataset.DatasetSchema()
        client.update_dataset_schema(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_dataset_schema_rest_bad_request(transport: str='rest', request_type=document_service.UpdateDatasetSchemaRequest):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset_schema': {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_dataset_schema(request)

def test_update_dataset_schema_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dataset.DatasetSchema()
        sample_request = {'dataset_schema': {'name': 'projects/sample1/locations/sample2/processors/sample3/dataset/datasetSchema'}}
        mock_args = dict(dataset_schema=dataset.DatasetSchema(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dataset.DatasetSchema.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_dataset_schema(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta3/{dataset_schema.name=projects/*/locations/*/processors/*/dataset/datasetSchema}' % client.transport._host, args[1])

def test_update_dataset_schema_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_dataset_schema(document_service.UpdateDatasetSchemaRequest(), dataset_schema=dataset.DatasetSchema(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_dataset_schema_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DocumentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DocumentServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DocumentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DocumentServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DocumentServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DocumentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DocumentServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.DocumentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DocumentServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.DocumentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DocumentServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DocumentServiceGrpcTransport, transports.DocumentServiceGrpcAsyncIOTransport, transports.DocumentServiceRestTransport])
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
        i = 10
        return i + 15
    transport = DocumentServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DocumentServiceGrpcTransport)

def test_document_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DocumentServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_document_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.documentai_v1beta3.services.document_service.transports.DocumentServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DocumentServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('update_dataset', 'import_documents', 'get_document', 'list_documents', 'batch_delete_documents', 'get_dataset_schema', 'update_dataset_schema', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_document_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.documentai_v1beta3.services.document_service.transports.DocumentServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DocumentServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_document_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.documentai_v1beta3.services.document_service.transports.DocumentServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DocumentServiceTransport()
        adc.assert_called_once()

def test_document_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DocumentServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DocumentServiceGrpcTransport, transports.DocumentServiceGrpcAsyncIOTransport])
def test_document_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DocumentServiceGrpcTransport, transports.DocumentServiceGrpcAsyncIOTransport, transports.DocumentServiceRestTransport])
def test_document_service_transport_auth_gdch_credentials(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DocumentServiceGrpcTransport, grpc_helpers), (transports.DocumentServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_document_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('documentai.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='documentai.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DocumentServiceGrpcTransport, transports.DocumentServiceGrpcAsyncIOTransport])
def test_document_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_document_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.DocumentServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_document_service_rest_lro_client():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_document_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='documentai.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('documentai.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_document_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='documentai.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('documentai.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_document_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = DocumentServiceClient(credentials=creds1, transport=transport_name)
    client2 = DocumentServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.update_dataset._session
    session2 = client2.transport.update_dataset._session
    assert session1 != session2
    session1 = client1.transport.import_documents._session
    session2 = client2.transport.import_documents._session
    assert session1 != session2
    session1 = client1.transport.get_document._session
    session2 = client2.transport.get_document._session
    assert session1 != session2
    session1 = client1.transport.list_documents._session
    session2 = client2.transport.list_documents._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_documents._session
    session2 = client2.transport.batch_delete_documents._session
    assert session1 != session2
    session1 = client1.transport.get_dataset_schema._session
    session2 = client2.transport.get_dataset_schema._session
    assert session1 != session2
    session1 = client1.transport.update_dataset_schema._session
    session2 = client2.transport.update_dataset_schema._session
    assert session1 != session2

def test_document_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DocumentServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_document_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DocumentServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DocumentServiceGrpcTransport, transports.DocumentServiceGrpcAsyncIOTransport])
def test_document_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.DocumentServiceGrpcTransport, transports.DocumentServiceGrpcAsyncIOTransport])
def test_document_service_transport_channel_mtls_with_adc(transport_class):
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

def test_document_service_grpc_lro_client():
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_document_service_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_dataset_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    processor = 'whelk'
    expected = 'projects/{project}/locations/{location}/processors/{processor}/dataset'.format(project=project, location=location, processor=processor)
    actual = DocumentServiceClient.dataset_path(project, location, processor)
    assert expected == actual

def test_parse_dataset_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'processor': 'nudibranch'}
    path = DocumentServiceClient.dataset_path(**expected)
    actual = DocumentServiceClient.parse_dataset_path(path)
    assert expected == actual

def test_dataset_schema_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    processor = 'winkle'
    expected = 'projects/{project}/locations/{location}/processors/{processor}/dataset/datasetSchema'.format(project=project, location=location, processor=processor)
    actual = DocumentServiceClient.dataset_schema_path(project, location, processor)
    assert expected == actual

def test_parse_dataset_schema_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'location': 'scallop', 'processor': 'abalone'}
    path = DocumentServiceClient.dataset_schema_path(**expected)
    actual = DocumentServiceClient.parse_dataset_schema_path(path)
    assert expected == actual

def test_schema_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    schema = 'whelk'
    expected = 'projects/{project}/locations/{location}/schemas/{schema}'.format(project=project, location=location, schema=schema)
    actual = DocumentServiceClient.schema_path(project, location, schema)
    assert expected == actual

def test_parse_schema_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'schema': 'nudibranch'}
    path = DocumentServiceClient.schema_path(**expected)
    actual = DocumentServiceClient.parse_schema_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DocumentServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = DocumentServiceClient.common_billing_account_path(**expected)
    actual = DocumentServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DocumentServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nautilus'}
    path = DocumentServiceClient.common_folder_path(**expected)
    actual = DocumentServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DocumentServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = DocumentServiceClient.common_organization_path(**expected)
    actual = DocumentServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = DocumentServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = DocumentServiceClient.common_project_path(**expected)
    actual = DocumentServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DocumentServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = DocumentServiceClient.common_location_path(**expected)
    actual = DocumentServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DocumentServiceTransport, '_prep_wrapped_messages') as prep:
        client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DocumentServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DocumentServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations'}, request)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations'}
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

def test_cancel_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = DocumentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = DocumentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DocumentServiceClient, transports.DocumentServiceGrpcTransport), (DocumentServiceAsyncClient, transports.DocumentServiceGrpcAsyncIOTransport)])
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