import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
from google.api import launch_stage_pb2
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
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.rpc import status_pb2
from google.type import color_pb2
from google.type import date_pb2
from google.type import datetime_pb2
from google.type import money_pb2
from google.type import postal_address_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.documentai_v1.services.document_processor_service import DocumentProcessorServiceAsyncClient, DocumentProcessorServiceClient, pagers, transports
from google.cloud.documentai_v1.types import barcode, document, document_io, document_processor_service, document_schema, evaluation, geometry
from google.cloud.documentai_v1.types import processor
from google.cloud.documentai_v1.types import processor as gcd_processor
from google.cloud.documentai_v1.types import processor_type

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
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
    assert DocumentProcessorServiceClient._get_default_mtls_endpoint(None) is None
    assert DocumentProcessorServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DocumentProcessorServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DocumentProcessorServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DocumentProcessorServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DocumentProcessorServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DocumentProcessorServiceClient, 'grpc'), (DocumentProcessorServiceAsyncClient, 'grpc_asyncio'), (DocumentProcessorServiceClient, 'rest')])
def test_document_processor_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('documentai.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DocumentProcessorServiceGrpcTransport, 'grpc'), (transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.DocumentProcessorServiceRestTransport, 'rest')])
def test_document_processor_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DocumentProcessorServiceClient, 'grpc'), (DocumentProcessorServiceAsyncClient, 'grpc_asyncio'), (DocumentProcessorServiceClient, 'rest')])
def test_document_processor_service_client_from_service_account_file(client_class, transport_name):
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

def test_document_processor_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = DocumentProcessorServiceClient.get_transport_class()
    available_transports = [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceRestTransport]
    assert transport in available_transports
    transport = DocumentProcessorServiceClient.get_transport_class('grpc')
    assert transport == transports.DocumentProcessorServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport, 'grpc'), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DocumentProcessorServiceClient, transports.DocumentProcessorServiceRestTransport, 'rest')])
@mock.patch.object(DocumentProcessorServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentProcessorServiceClient))
@mock.patch.object(DocumentProcessorServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentProcessorServiceAsyncClient))
def test_document_processor_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(DocumentProcessorServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DocumentProcessorServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport, 'grpc', 'true'), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport, 'grpc', 'false'), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (DocumentProcessorServiceClient, transports.DocumentProcessorServiceRestTransport, 'rest', 'true'), (DocumentProcessorServiceClient, transports.DocumentProcessorServiceRestTransport, 'rest', 'false')])
@mock.patch.object(DocumentProcessorServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentProcessorServiceClient))
@mock.patch.object(DocumentProcessorServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentProcessorServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_document_processor_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [DocumentProcessorServiceClient, DocumentProcessorServiceAsyncClient])
@mock.patch.object(DocumentProcessorServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentProcessorServiceClient))
@mock.patch.object(DocumentProcessorServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DocumentProcessorServiceAsyncClient))
def test_document_processor_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport, 'grpc'), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DocumentProcessorServiceClient, transports.DocumentProcessorServiceRestTransport, 'rest')])
def test_document_processor_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport, 'grpc', grpc_helpers), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (DocumentProcessorServiceClient, transports.DocumentProcessorServiceRestTransport, 'rest', None)])
def test_document_processor_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_document_processor_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.documentai_v1.services.document_processor_service.transports.DocumentProcessorServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DocumentProcessorServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport, 'grpc', grpc_helpers), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_document_processor_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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

@pytest.mark.parametrize('request_type', [document_processor_service.ProcessRequest, dict])
def test_process_document(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        call.return_value = document_processor_service.ProcessResponse()
        response = client.process_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ProcessRequest()
    assert isinstance(response, document_processor_service.ProcessResponse)

def test_process_document_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        client.process_document()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ProcessRequest()

@pytest.mark.asyncio
async def test_process_document_async(transport: str='grpc_asyncio', request_type=document_processor_service.ProcessRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ProcessResponse())
        response = await client.process_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ProcessRequest()
    assert isinstance(response, document_processor_service.ProcessResponse)

@pytest.mark.asyncio
async def test_process_document_async_from_dict():
    await test_process_document_async(request_type=dict)

def test_process_document_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ProcessRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        call.return_value = document_processor_service.ProcessResponse()
        client.process_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_process_document_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ProcessRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ProcessResponse())
        await client.process_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_process_document_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        call.return_value = document_processor_service.ProcessResponse()
        client.process_document(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_process_document_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.process_document(document_processor_service.ProcessRequest(), name='name_value')

@pytest.mark.asyncio
async def test_process_document_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.process_document), '__call__') as call:
        call.return_value = document_processor_service.ProcessResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ProcessResponse())
        response = await client.process_document(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_process_document_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.process_document(document_processor_service.ProcessRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.BatchProcessRequest, dict])
def test_batch_process_documents(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_process_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.BatchProcessRequest()
    assert isinstance(response, future.Future)

def test_batch_process_documents_empty_call():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        client.batch_process_documents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.BatchProcessRequest()

@pytest.mark.asyncio
async def test_batch_process_documents_async(transport: str='grpc_asyncio', request_type=document_processor_service.BatchProcessRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_process_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.BatchProcessRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_process_documents_async_from_dict():
    await test_batch_process_documents_async(request_type=dict)

def test_batch_process_documents_field_headers():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.BatchProcessRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_process_documents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_process_documents_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.BatchProcessRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_process_documents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_batch_process_documents_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_process_documents(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_batch_process_documents_flattened_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_process_documents(document_processor_service.BatchProcessRequest(), name='name_value')

@pytest.mark.asyncio
async def test_batch_process_documents_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_process_documents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_process_documents(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_process_documents_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_process_documents(document_processor_service.BatchProcessRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.FetchProcessorTypesRequest, dict])
def test_fetch_processor_types(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        call.return_value = document_processor_service.FetchProcessorTypesResponse()
        response = client.fetch_processor_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.FetchProcessorTypesRequest()
    assert isinstance(response, document_processor_service.FetchProcessorTypesResponse)

def test_fetch_processor_types_empty_call():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        client.fetch_processor_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.FetchProcessorTypesRequest()

@pytest.mark.asyncio
async def test_fetch_processor_types_async(transport: str='grpc_asyncio', request_type=document_processor_service.FetchProcessorTypesRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.FetchProcessorTypesResponse())
        response = await client.fetch_processor_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.FetchProcessorTypesRequest()
    assert isinstance(response, document_processor_service.FetchProcessorTypesResponse)

@pytest.mark.asyncio
async def test_fetch_processor_types_async_from_dict():
    await test_fetch_processor_types_async(request_type=dict)

def test_fetch_processor_types_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.FetchProcessorTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        call.return_value = document_processor_service.FetchProcessorTypesResponse()
        client.fetch_processor_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_processor_types_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.FetchProcessorTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.FetchProcessorTypesResponse())
        await client.fetch_processor_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_fetch_processor_types_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        call.return_value = document_processor_service.FetchProcessorTypesResponse()
        client.fetch_processor_types(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_fetch_processor_types_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.fetch_processor_types(document_processor_service.FetchProcessorTypesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_fetch_processor_types_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_processor_types), '__call__') as call:
        call.return_value = document_processor_service.FetchProcessorTypesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.FetchProcessorTypesResponse())
        response = await client.fetch_processor_types(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_fetch_processor_types_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.fetch_processor_types(document_processor_service.FetchProcessorTypesRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [document_processor_service.ListProcessorTypesRequest, dict])
def test_list_processor_types(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorTypesResponse(next_page_token='next_page_token_value')
        response = client.list_processor_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorTypesRequest()
    assert isinstance(response, pagers.ListProcessorTypesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_processor_types_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        client.list_processor_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorTypesRequest()

@pytest.mark.asyncio
async def test_list_processor_types_async(transport: str='grpc_asyncio', request_type=document_processor_service.ListProcessorTypesRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorTypesResponse(next_page_token='next_page_token_value'))
        response = await client.list_processor_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorTypesRequest()
    assert isinstance(response, pagers.ListProcessorTypesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_processor_types_async_from_dict():
    await test_list_processor_types_async(request_type=dict)

def test_list_processor_types_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListProcessorTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorTypesResponse()
        client.list_processor_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_processor_types_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListProcessorTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorTypesResponse())
        await client.list_processor_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_processor_types_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorTypesResponse()
        client.list_processor_types(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_processor_types_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_processor_types(document_processor_service.ListProcessorTypesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_processor_types_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorTypesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorTypesResponse())
        response = await client.list_processor_types(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_processor_types_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_processor_types(document_processor_service.ListProcessorTypesRequest(), parent='parent_value')

def test_list_processor_types_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.side_effect = (document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType(), processor_type.ProcessorType()], next_page_token='abc'), document_processor_service.ListProcessorTypesResponse(processor_types=[], next_page_token='def'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType()], next_page_token='ghi'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_processor_types(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, processor_type.ProcessorType) for i in results))

def test_list_processor_types_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_processor_types), '__call__') as call:
        call.side_effect = (document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType(), processor_type.ProcessorType()], next_page_token='abc'), document_processor_service.ListProcessorTypesResponse(processor_types=[], next_page_token='def'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType()], next_page_token='ghi'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType()]), RuntimeError)
        pages = list(client.list_processor_types(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_processor_types_async_pager():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_processor_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType(), processor_type.ProcessorType()], next_page_token='abc'), document_processor_service.ListProcessorTypesResponse(processor_types=[], next_page_token='def'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType()], next_page_token='ghi'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType()]), RuntimeError)
        async_pager = await client.list_processor_types(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, processor_type.ProcessorType) for i in responses))

@pytest.mark.asyncio
async def test_list_processor_types_async_pages():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_processor_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType(), processor_type.ProcessorType()], next_page_token='abc'), document_processor_service.ListProcessorTypesResponse(processor_types=[], next_page_token='def'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType()], next_page_token='ghi'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_processor_types(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.GetProcessorTypeRequest, dict])
def test_get_processor_type(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        call.return_value = processor_type.ProcessorType(name='name_value', type_='type__value', category='category_value', allow_creation=True, launch_stage=launch_stage_pb2.LaunchStage.UNIMPLEMENTED, sample_document_uris=['sample_document_uris_value'])
        response = client.get_processor_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorTypeRequest()
    assert isinstance(response, processor_type.ProcessorType)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.category == 'category_value'
    assert response.allow_creation is True
    assert response.launch_stage == launch_stage_pb2.LaunchStage.UNIMPLEMENTED
    assert response.sample_document_uris == ['sample_document_uris_value']

def test_get_processor_type_empty_call():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        client.get_processor_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorTypeRequest()

@pytest.mark.asyncio
async def test_get_processor_type_async(transport: str='grpc_asyncio', request_type=document_processor_service.GetProcessorTypeRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor_type.ProcessorType(name='name_value', type_='type__value', category='category_value', allow_creation=True, launch_stage=launch_stage_pb2.LaunchStage.UNIMPLEMENTED, sample_document_uris=['sample_document_uris_value']))
        response = await client.get_processor_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorTypeRequest()
    assert isinstance(response, processor_type.ProcessorType)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.category == 'category_value'
    assert response.allow_creation is True
    assert response.launch_stage == launch_stage_pb2.LaunchStage.UNIMPLEMENTED
    assert response.sample_document_uris == ['sample_document_uris_value']

@pytest.mark.asyncio
async def test_get_processor_type_async_from_dict():
    await test_get_processor_type_async(request_type=dict)

def test_get_processor_type_field_headers():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetProcessorTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        call.return_value = processor_type.ProcessorType()
        client.get_processor_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_processor_type_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetProcessorTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor_type.ProcessorType())
        await client.get_processor_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_processor_type_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        call.return_value = processor_type.ProcessorType()
        client.get_processor_type(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_processor_type_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_processor_type(document_processor_service.GetProcessorTypeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_processor_type_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_processor_type), '__call__') as call:
        call.return_value = processor_type.ProcessorType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor_type.ProcessorType())
        response = await client.get_processor_type(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_processor_type_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_processor_type(document_processor_service.GetProcessorTypeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.ListProcessorsRequest, dict])
def test_list_processors(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorsResponse(next_page_token='next_page_token_value')
        response = client.list_processors(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorsRequest()
    assert isinstance(response, pagers.ListProcessorsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_processors_empty_call():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        client.list_processors()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorsRequest()

@pytest.mark.asyncio
async def test_list_processors_async(transport: str='grpc_asyncio', request_type=document_processor_service.ListProcessorsRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorsResponse(next_page_token='next_page_token_value'))
        response = await client.list_processors(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorsRequest()
    assert isinstance(response, pagers.ListProcessorsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_processors_async_from_dict():
    await test_list_processors_async(request_type=dict)

def test_list_processors_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListProcessorsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorsResponse()
        client.list_processors(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_processors_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListProcessorsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorsResponse())
        await client.list_processors(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_processors_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorsResponse()
        client.list_processors(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_processors_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_processors(document_processor_service.ListProcessorsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_processors_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorsResponse())
        response = await client.list_processors(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_processors_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_processors(document_processor_service.ListProcessorsRequest(), parent='parent_value')

def test_list_processors_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.side_effect = (document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor(), processor.Processor()], next_page_token='abc'), document_processor_service.ListProcessorsResponse(processors=[], next_page_token='def'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor()], next_page_token='ghi'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_processors(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, processor.Processor) for i in results))

def test_list_processors_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_processors), '__call__') as call:
        call.side_effect = (document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor(), processor.Processor()], next_page_token='abc'), document_processor_service.ListProcessorsResponse(processors=[], next_page_token='def'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor()], next_page_token='ghi'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor()]), RuntimeError)
        pages = list(client.list_processors(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_processors_async_pager():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_processors), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor(), processor.Processor()], next_page_token='abc'), document_processor_service.ListProcessorsResponse(processors=[], next_page_token='def'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor()], next_page_token='ghi'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor()]), RuntimeError)
        async_pager = await client.list_processors(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, processor.Processor) for i in responses))

@pytest.mark.asyncio
async def test_list_processors_async_pages():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_processors), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor(), processor.Processor()], next_page_token='abc'), document_processor_service.ListProcessorsResponse(processors=[], next_page_token='def'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor()], next_page_token='ghi'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_processors(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.GetProcessorRequest, dict])
def test_get_processor(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        call.return_value = processor.Processor(name='name_value', type_='type__value', display_name='display_name_value', state=processor.Processor.State.ENABLED, default_processor_version='default_processor_version_value', process_endpoint='process_endpoint_value', kms_key_name='kms_key_name_value')
        response = client.get_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorRequest()
    assert isinstance(response, processor.Processor)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.display_name == 'display_name_value'
    assert response.state == processor.Processor.State.ENABLED
    assert response.default_processor_version == 'default_processor_version_value'
    assert response.process_endpoint == 'process_endpoint_value'
    assert response.kms_key_name == 'kms_key_name_value'

def test_get_processor_empty_call():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        client.get_processor()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorRequest()

@pytest.mark.asyncio
async def test_get_processor_async(transport: str='grpc_asyncio', request_type=document_processor_service.GetProcessorRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor.Processor(name='name_value', type_='type__value', display_name='display_name_value', state=processor.Processor.State.ENABLED, default_processor_version='default_processor_version_value', process_endpoint='process_endpoint_value', kms_key_name='kms_key_name_value'))
        response = await client.get_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorRequest()
    assert isinstance(response, processor.Processor)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.display_name == 'display_name_value'
    assert response.state == processor.Processor.State.ENABLED
    assert response.default_processor_version == 'default_processor_version_value'
    assert response.process_endpoint == 'process_endpoint_value'
    assert response.kms_key_name == 'kms_key_name_value'

@pytest.mark.asyncio
async def test_get_processor_async_from_dict():
    await test_get_processor_async(request_type=dict)

def test_get_processor_field_headers():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        call.return_value = processor.Processor()
        client.get_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_processor_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor.Processor())
        await client.get_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_processor_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        call.return_value = processor.Processor()
        client.get_processor(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_processor_flattened_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_processor(document_processor_service.GetProcessorRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_processor_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_processor), '__call__') as call:
        call.return_value = processor.Processor()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor.Processor())
        response = await client.get_processor(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_processor_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_processor(document_processor_service.GetProcessorRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.TrainProcessorVersionRequest, dict])
def test_train_processor_version(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.train_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.TrainProcessorVersionRequest()
    assert isinstance(response, future.Future)

def test_train_processor_version_empty_call():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        client.train_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.TrainProcessorVersionRequest()

@pytest.mark.asyncio
async def test_train_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.TrainProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.train_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.TrainProcessorVersionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_train_processor_version_async_from_dict():
    await test_train_processor_version_async(request_type=dict)

def test_train_processor_version_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.TrainProcessorVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.train_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_train_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.TrainProcessorVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.train_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_train_processor_version_flattened():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.train_processor_version(parent='parent_value', processor_version=processor.ProcessorVersion(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].processor_version
        mock_val = processor.ProcessorVersion(name='name_value')
        assert arg == mock_val

def test_train_processor_version_flattened_error():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.train_processor_version(document_processor_service.TrainProcessorVersionRequest(), parent='parent_value', processor_version=processor.ProcessorVersion(name='name_value'))

@pytest.mark.asyncio
async def test_train_processor_version_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.train_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.train_processor_version(parent='parent_value', processor_version=processor.ProcessorVersion(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].processor_version
        mock_val = processor.ProcessorVersion(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_train_processor_version_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.train_processor_version(document_processor_service.TrainProcessorVersionRequest(), parent='parent_value', processor_version=processor.ProcessorVersion(name='name_value'))

@pytest.mark.parametrize('request_type', [document_processor_service.GetProcessorVersionRequest, dict])
def test_get_processor_version(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        call.return_value = processor.ProcessorVersion(name='name_value', display_name='display_name_value', state=processor.ProcessorVersion.State.DEPLOYED, kms_key_name='kms_key_name_value', kms_key_version_name='kms_key_version_name_value', google_managed=True)
        response = client.get_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorVersionRequest()
    assert isinstance(response, processor.ProcessorVersion)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == processor.ProcessorVersion.State.DEPLOYED
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.kms_key_version_name == 'kms_key_version_name_value'
    assert response.google_managed is True

def test_get_processor_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        client.get_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorVersionRequest()

@pytest.mark.asyncio
async def test_get_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.GetProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor.ProcessorVersion(name='name_value', display_name='display_name_value', state=processor.ProcessorVersion.State.DEPLOYED, kms_key_name='kms_key_name_value', kms_key_version_name='kms_key_version_name_value', google_managed=True))
        response = await client.get_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetProcessorVersionRequest()
    assert isinstance(response, processor.ProcessorVersion)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == processor.ProcessorVersion.State.DEPLOYED
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.kms_key_version_name == 'kms_key_version_name_value'
    assert response.google_managed is True

@pytest.mark.asyncio
async def test_get_processor_version_async_from_dict():
    await test_get_processor_version_async(request_type=dict)

def test_get_processor_version_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        call.return_value = processor.ProcessorVersion()
        client.get_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor.ProcessorVersion())
        await client.get_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_processor_version_flattened():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        call.return_value = processor.ProcessorVersion()
        client.get_processor_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_processor_version_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_processor_version(document_processor_service.GetProcessorVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_processor_version_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_processor_version), '__call__') as call:
        call.return_value = processor.ProcessorVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(processor.ProcessorVersion())
        response = await client.get_processor_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_processor_version_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_processor_version(document_processor_service.GetProcessorVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.ListProcessorVersionsRequest, dict])
def test_list_processor_versions(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorVersionsResponse(next_page_token='next_page_token_value')
        response = client.list_processor_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorVersionsRequest()
    assert isinstance(response, pagers.ListProcessorVersionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_processor_versions_empty_call():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        client.list_processor_versions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorVersionsRequest()

@pytest.mark.asyncio
async def test_list_processor_versions_async(transport: str='grpc_asyncio', request_type=document_processor_service.ListProcessorVersionsRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorVersionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_processor_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListProcessorVersionsRequest()
    assert isinstance(response, pagers.ListProcessorVersionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_processor_versions_async_from_dict():
    await test_list_processor_versions_async(request_type=dict)

def test_list_processor_versions_field_headers():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListProcessorVersionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorVersionsResponse()
        client.list_processor_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_processor_versions_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListProcessorVersionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorVersionsResponse())
        await client.list_processor_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_processor_versions_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorVersionsResponse()
        client.list_processor_versions(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_processor_versions_flattened_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_processor_versions(document_processor_service.ListProcessorVersionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_processor_versions_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.return_value = document_processor_service.ListProcessorVersionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListProcessorVersionsResponse())
        response = await client.list_processor_versions(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_processor_versions_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_processor_versions(document_processor_service.ListProcessorVersionsRequest(), parent='parent_value')

def test_list_processor_versions_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.side_effect = (document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion(), processor.ProcessorVersion()], next_page_token='abc'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[], next_page_token='def'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion()], next_page_token='ghi'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_processor_versions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, processor.ProcessorVersion) for i in results))

def test_list_processor_versions_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__') as call:
        call.side_effect = (document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion(), processor.ProcessorVersion()], next_page_token='abc'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[], next_page_token='def'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion()], next_page_token='ghi'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion()]), RuntimeError)
        pages = list(client.list_processor_versions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_processor_versions_async_pager():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion(), processor.ProcessorVersion()], next_page_token='abc'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[], next_page_token='def'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion()], next_page_token='ghi'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion()]), RuntimeError)
        async_pager = await client.list_processor_versions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, processor.ProcessorVersion) for i in responses))

@pytest.mark.asyncio
async def test_list_processor_versions_async_pages():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_processor_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion(), processor.ProcessorVersion()], next_page_token='abc'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[], next_page_token='def'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion()], next_page_token='ghi'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_processor_versions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.DeleteProcessorVersionRequest, dict])
def test_delete_processor_version(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeleteProcessorVersionRequest()
    assert isinstance(response, future.Future)

def test_delete_processor_version_empty_call():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        client.delete_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeleteProcessorVersionRequest()

@pytest.mark.asyncio
async def test_delete_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.DeleteProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeleteProcessorVersionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_processor_version_async_from_dict():
    await test_delete_processor_version_async(request_type=dict)

def test_delete_processor_version_field_headers():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DeleteProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DeleteProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_processor_version_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_processor_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_processor_version_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_processor_version(document_processor_service.DeleteProcessorVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_processor_version_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_processor_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_processor_version_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_processor_version(document_processor_service.DeleteProcessorVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.DeployProcessorVersionRequest, dict])
def test_deploy_processor_version(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.deploy_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeployProcessorVersionRequest()
    assert isinstance(response, future.Future)

def test_deploy_processor_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        client.deploy_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeployProcessorVersionRequest()

@pytest.mark.asyncio
async def test_deploy_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.DeployProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.deploy_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeployProcessorVersionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_deploy_processor_version_async_from_dict():
    await test_deploy_processor_version_async(request_type=dict)

def test_deploy_processor_version_field_headers():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DeployProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.deploy_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_deploy_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DeployProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.deploy_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_deploy_processor_version_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.deploy_processor_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_deploy_processor_version_flattened_error():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.deploy_processor_version(document_processor_service.DeployProcessorVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_deploy_processor_version_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.deploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.deploy_processor_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_deploy_processor_version_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.deploy_processor_version(document_processor_service.DeployProcessorVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.UndeployProcessorVersionRequest, dict])
def test_undeploy_processor_version(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.undeploy_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.UndeployProcessorVersionRequest()
    assert isinstance(response, future.Future)

def test_undeploy_processor_version_empty_call():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        client.undeploy_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.UndeployProcessorVersionRequest()

@pytest.mark.asyncio
async def test_undeploy_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.UndeployProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undeploy_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.UndeployProcessorVersionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_undeploy_processor_version_async_from_dict():
    await test_undeploy_processor_version_async(request_type=dict)

def test_undeploy_processor_version_field_headers():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.UndeployProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undeploy_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_undeploy_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.UndeployProcessorVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.undeploy_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_undeploy_processor_version_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undeploy_processor_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_undeploy_processor_version_flattened_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.undeploy_processor_version(document_processor_service.UndeployProcessorVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_undeploy_processor_version_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undeploy_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undeploy_processor_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_undeploy_processor_version_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.undeploy_processor_version(document_processor_service.UndeployProcessorVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.CreateProcessorRequest, dict])
def test_create_processor(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        call.return_value = gcd_processor.Processor(name='name_value', type_='type__value', display_name='display_name_value', state=gcd_processor.Processor.State.ENABLED, default_processor_version='default_processor_version_value', process_endpoint='process_endpoint_value', kms_key_name='kms_key_name_value')
        response = client.create_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.CreateProcessorRequest()
    assert isinstance(response, gcd_processor.Processor)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.display_name == 'display_name_value'
    assert response.state == gcd_processor.Processor.State.ENABLED
    assert response.default_processor_version == 'default_processor_version_value'
    assert response.process_endpoint == 'process_endpoint_value'
    assert response.kms_key_name == 'kms_key_name_value'

def test_create_processor_empty_call():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        client.create_processor()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.CreateProcessorRequest()

@pytest.mark.asyncio
async def test_create_processor_async(transport: str='grpc_asyncio', request_type=document_processor_service.CreateProcessorRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_processor.Processor(name='name_value', type_='type__value', display_name='display_name_value', state=gcd_processor.Processor.State.ENABLED, default_processor_version='default_processor_version_value', process_endpoint='process_endpoint_value', kms_key_name='kms_key_name_value'))
        response = await client.create_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.CreateProcessorRequest()
    assert isinstance(response, gcd_processor.Processor)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.display_name == 'display_name_value'
    assert response.state == gcd_processor.Processor.State.ENABLED
    assert response.default_processor_version == 'default_processor_version_value'
    assert response.process_endpoint == 'process_endpoint_value'
    assert response.kms_key_name == 'kms_key_name_value'

@pytest.mark.asyncio
async def test_create_processor_async_from_dict():
    await test_create_processor_async(request_type=dict)

def test_create_processor_field_headers():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.CreateProcessorRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        call.return_value = gcd_processor.Processor()
        client.create_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_processor_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.CreateProcessorRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_processor.Processor())
        await client.create_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_processor_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        call.return_value = gcd_processor.Processor()
        client.create_processor(parent='parent_value', processor=gcd_processor.Processor(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].processor
        mock_val = gcd_processor.Processor(name='name_value')
        assert arg == mock_val

def test_create_processor_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_processor(document_processor_service.CreateProcessorRequest(), parent='parent_value', processor=gcd_processor.Processor(name='name_value'))

@pytest.mark.asyncio
async def test_create_processor_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_processor), '__call__') as call:
        call.return_value = gcd_processor.Processor()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_processor.Processor())
        response = await client.create_processor(parent='parent_value', processor=gcd_processor.Processor(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].processor
        mock_val = gcd_processor.Processor(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_processor_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_processor(document_processor_service.CreateProcessorRequest(), parent='parent_value', processor=gcd_processor.Processor(name='name_value'))

@pytest.mark.parametrize('request_type', [document_processor_service.DeleteProcessorRequest, dict])
def test_delete_processor(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeleteProcessorRequest()
    assert isinstance(response, future.Future)

def test_delete_processor_empty_call():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        client.delete_processor()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeleteProcessorRequest()

@pytest.mark.asyncio
async def test_delete_processor_async(transport: str='grpc_asyncio', request_type=document_processor_service.DeleteProcessorRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DeleteProcessorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_processor_async_from_dict():
    await test_delete_processor_async(request_type=dict)

def test_delete_processor_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DeleteProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_processor_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DeleteProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_processor_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_processor(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_processor_flattened_error():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_processor(document_processor_service.DeleteProcessorRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_processor_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_processor(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_processor_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_processor(document_processor_service.DeleteProcessorRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.EnableProcessorRequest, dict])
def test_enable_processor(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.enable_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.EnableProcessorRequest()
    assert isinstance(response, future.Future)

def test_enable_processor_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.enable_processor), '__call__') as call:
        client.enable_processor()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.EnableProcessorRequest()

@pytest.mark.asyncio
async def test_enable_processor_async(transport: str='grpc_asyncio', request_type=document_processor_service.EnableProcessorRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.enable_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.EnableProcessorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_enable_processor_async_from_dict():
    await test_enable_processor_async(request_type=dict)

def test_enable_processor_field_headers():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.EnableProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.enable_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_enable_processor_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.EnableProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.enable_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [document_processor_service.DisableProcessorRequest, dict])
def test_disable_processor(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.disable_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DisableProcessorRequest()
    assert isinstance(response, future.Future)

def test_disable_processor_empty_call():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.disable_processor), '__call__') as call:
        client.disable_processor()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DisableProcessorRequest()

@pytest.mark.asyncio
async def test_disable_processor_async(transport: str='grpc_asyncio', request_type=document_processor_service.DisableProcessorRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.disable_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.DisableProcessorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_disable_processor_async_from_dict():
    await test_disable_processor_async(request_type=dict)

def test_disable_processor_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DisableProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_processor), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.disable_processor(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_disable_processor_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.DisableProcessorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_processor), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.disable_processor(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [document_processor_service.SetDefaultProcessorVersionRequest, dict])
def test_set_default_processor_version(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_default_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.set_default_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.SetDefaultProcessorVersionRequest()
    assert isinstance(response, future.Future)

def test_set_default_processor_version_empty_call():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_default_processor_version), '__call__') as call:
        client.set_default_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.SetDefaultProcessorVersionRequest()

@pytest.mark.asyncio
async def test_set_default_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.SetDefaultProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_default_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.set_default_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.SetDefaultProcessorVersionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_set_default_processor_version_async_from_dict():
    await test_set_default_processor_version_async(request_type=dict)

def test_set_default_processor_version_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.SetDefaultProcessorVersionRequest()
    request.processor = 'processor_value'
    with mock.patch.object(type(client.transport.set_default_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.set_default_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'processor=processor_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_default_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.SetDefaultProcessorVersionRequest()
    request.processor = 'processor_value'
    with mock.patch.object(type(client.transport.set_default_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.set_default_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'processor=processor_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [document_processor_service.ReviewDocumentRequest, dict])
def test_review_document(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.review_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ReviewDocumentRequest()
    assert isinstance(response, future.Future)

def test_review_document_empty_call():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        client.review_document()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ReviewDocumentRequest()

@pytest.mark.asyncio
async def test_review_document_async(transport: str='grpc_asyncio', request_type=document_processor_service.ReviewDocumentRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.review_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ReviewDocumentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_review_document_async_from_dict():
    await test_review_document_async(request_type=dict)

def test_review_document_field_headers():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ReviewDocumentRequest()
    request.human_review_config = 'human_review_config_value'
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.review_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'human_review_config=human_review_config_value') in kw['metadata']

@pytest.mark.asyncio
async def test_review_document_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ReviewDocumentRequest()
    request.human_review_config = 'human_review_config_value'
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.review_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'human_review_config=human_review_config_value') in kw['metadata']

def test_review_document_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.review_document(human_review_config='human_review_config_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].human_review_config
        mock_val = 'human_review_config_value'
        assert arg == mock_val

def test_review_document_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.review_document(document_processor_service.ReviewDocumentRequest(), human_review_config='human_review_config_value')

@pytest.mark.asyncio
async def test_review_document_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.review_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.review_document(human_review_config='human_review_config_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].human_review_config
        mock_val = 'human_review_config_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_review_document_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.review_document(document_processor_service.ReviewDocumentRequest(), human_review_config='human_review_config_value')

@pytest.mark.parametrize('request_type', [document_processor_service.EvaluateProcessorVersionRequest, dict])
def test_evaluate_processor_version(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.evaluate_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.EvaluateProcessorVersionRequest()
    assert isinstance(response, future.Future)

def test_evaluate_processor_version_empty_call():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        client.evaluate_processor_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.EvaluateProcessorVersionRequest()

@pytest.mark.asyncio
async def test_evaluate_processor_version_async(transport: str='grpc_asyncio', request_type=document_processor_service.EvaluateProcessorVersionRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.evaluate_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.EvaluateProcessorVersionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_evaluate_processor_version_async_from_dict():
    await test_evaluate_processor_version_async(request_type=dict)

def test_evaluate_processor_version_field_headers():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.EvaluateProcessorVersionRequest()
    request.processor_version = 'processor_version_value'
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.evaluate_processor_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'processor_version=processor_version_value') in kw['metadata']

@pytest.mark.asyncio
async def test_evaluate_processor_version_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.EvaluateProcessorVersionRequest()
    request.processor_version = 'processor_version_value'
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.evaluate_processor_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'processor_version=processor_version_value') in kw['metadata']

def test_evaluate_processor_version_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.evaluate_processor_version(processor_version='processor_version_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].processor_version
        mock_val = 'processor_version_value'
        assert arg == mock_val

def test_evaluate_processor_version_flattened_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.evaluate_processor_version(document_processor_service.EvaluateProcessorVersionRequest(), processor_version='processor_version_value')

@pytest.mark.asyncio
async def test_evaluate_processor_version_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.evaluate_processor_version), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.evaluate_processor_version(processor_version='processor_version_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].processor_version
        mock_val = 'processor_version_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_evaluate_processor_version_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.evaluate_processor_version(document_processor_service.EvaluateProcessorVersionRequest(), processor_version='processor_version_value')

@pytest.mark.parametrize('request_type', [document_processor_service.GetEvaluationRequest, dict])
def test_get_evaluation(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        call.return_value = evaluation.Evaluation(name='name_value', kms_key_name='kms_key_name_value', kms_key_version_name='kms_key_version_name_value')
        response = client.get_evaluation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetEvaluationRequest()
    assert isinstance(response, evaluation.Evaluation)
    assert response.name == 'name_value'
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.kms_key_version_name == 'kms_key_version_name_value'

def test_get_evaluation_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        client.get_evaluation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetEvaluationRequest()

@pytest.mark.asyncio
async def test_get_evaluation_async(transport: str='grpc_asyncio', request_type=document_processor_service.GetEvaluationRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(evaluation.Evaluation(name='name_value', kms_key_name='kms_key_name_value', kms_key_version_name='kms_key_version_name_value'))
        response = await client.get_evaluation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.GetEvaluationRequest()
    assert isinstance(response, evaluation.Evaluation)
    assert response.name == 'name_value'
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.kms_key_version_name == 'kms_key_version_name_value'

@pytest.mark.asyncio
async def test_get_evaluation_async_from_dict():
    await test_get_evaluation_async(request_type=dict)

def test_get_evaluation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetEvaluationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        call.return_value = evaluation.Evaluation()
        client.get_evaluation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_evaluation_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.GetEvaluationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(evaluation.Evaluation())
        await client.get_evaluation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_evaluation_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        call.return_value = evaluation.Evaluation()
        client.get_evaluation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_evaluation_flattened_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_evaluation(document_processor_service.GetEvaluationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_evaluation_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_evaluation), '__call__') as call:
        call.return_value = evaluation.Evaluation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(evaluation.Evaluation())
        response = await client.get_evaluation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_evaluation_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_evaluation(document_processor_service.GetEvaluationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [document_processor_service.ListEvaluationsRequest, dict])
def test_list_evaluations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.return_value = document_processor_service.ListEvaluationsResponse(next_page_token='next_page_token_value')
        response = client.list_evaluations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListEvaluationsRequest()
    assert isinstance(response, pagers.ListEvaluationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_evaluations_empty_call():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        client.list_evaluations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListEvaluationsRequest()

@pytest.mark.asyncio
async def test_list_evaluations_async(transport: str='grpc_asyncio', request_type=document_processor_service.ListEvaluationsRequest):
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListEvaluationsResponse(next_page_token='next_page_token_value'))
        response = await client.list_evaluations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == document_processor_service.ListEvaluationsRequest()
    assert isinstance(response, pagers.ListEvaluationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_evaluations_async_from_dict():
    await test_list_evaluations_async(request_type=dict)

def test_list_evaluations_field_headers():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListEvaluationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.return_value = document_processor_service.ListEvaluationsResponse()
        client.list_evaluations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_evaluations_field_headers_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = document_processor_service.ListEvaluationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListEvaluationsResponse())
        await client.list_evaluations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_evaluations_flattened():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.return_value = document_processor_service.ListEvaluationsResponse()
        client.list_evaluations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_evaluations_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_evaluations(document_processor_service.ListEvaluationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_evaluations_flattened_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.return_value = document_processor_service.ListEvaluationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(document_processor_service.ListEvaluationsResponse())
        response = await client.list_evaluations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_evaluations_flattened_error_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_evaluations(document_processor_service.ListEvaluationsRequest(), parent='parent_value')

def test_list_evaluations_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.side_effect = (document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation(), evaluation.Evaluation()], next_page_token='abc'), document_processor_service.ListEvaluationsResponse(evaluations=[], next_page_token='def'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation()], next_page_token='ghi'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_evaluations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, evaluation.Evaluation) for i in results))

def test_list_evaluations_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_evaluations), '__call__') as call:
        call.side_effect = (document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation(), evaluation.Evaluation()], next_page_token='abc'), document_processor_service.ListEvaluationsResponse(evaluations=[], next_page_token='def'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation()], next_page_token='ghi'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation()]), RuntimeError)
        pages = list(client.list_evaluations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_evaluations_async_pager():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_evaluations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation(), evaluation.Evaluation()], next_page_token='abc'), document_processor_service.ListEvaluationsResponse(evaluations=[], next_page_token='def'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation()], next_page_token='ghi'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation()]), RuntimeError)
        async_pager = await client.list_evaluations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, evaluation.Evaluation) for i in responses))

@pytest.mark.asyncio
async def test_list_evaluations_async_pages():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_evaluations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation(), evaluation.Evaluation()], next_page_token='abc'), document_processor_service.ListEvaluationsResponse(evaluations=[], next_page_token='def'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation()], next_page_token='ghi'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_evaluations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.ProcessRequest, dict])
def test_process_document_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ProcessResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ProcessResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.process_document(request)
    assert isinstance(response, document_processor_service.ProcessResponse)

def test_process_document_rest_required_fields(request_type=document_processor_service.ProcessRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).process_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).process_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_processor_service.ProcessResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_processor_service.ProcessResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.process_document(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_process_document_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.process_document._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_process_document_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_process_document') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_process_document') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.ProcessRequest.pb(document_processor_service.ProcessRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_processor_service.ProcessResponse.to_json(document_processor_service.ProcessResponse())
        request = document_processor_service.ProcessRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_processor_service.ProcessResponse()
        client.process_document(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_process_document_rest_bad_request(transport: str='rest', request_type=document_processor_service.ProcessRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.process_document(request)

def test_process_document_rest_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ProcessResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ProcessResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.process_document(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*}:process' % client.transport._host, args[1])

def test_process_document_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.process_document(document_processor_service.ProcessRequest(), name='name_value')

def test_process_document_rest_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.BatchProcessRequest, dict])
def test_batch_process_documents_rest(request_type):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_process_documents(request)
    assert response.operation.name == 'operations/spam'

def test_batch_process_documents_rest_required_fields(request_type=document_processor_service.BatchProcessRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_process_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_process_documents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_process_documents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_process_documents_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_process_documents._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_process_documents_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_batch_process_documents') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_batch_process_documents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.BatchProcessRequest.pb(document_processor_service.BatchProcessRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.BatchProcessRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_process_documents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_process_documents_rest_bad_request(transport: str='rest', request_type=document_processor_service.BatchProcessRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_process_documents(request)

def test_batch_process_documents_rest_flattened():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_process_documents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*}:batchProcess' % client.transport._host, args[1])

def test_batch_process_documents_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_process_documents(document_processor_service.BatchProcessRequest(), name='name_value')

def test_batch_process_documents_rest_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.FetchProcessorTypesRequest, dict])
def test_fetch_processor_types_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.FetchProcessorTypesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.FetchProcessorTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_processor_types(request)
    assert isinstance(response, document_processor_service.FetchProcessorTypesResponse)

def test_fetch_processor_types_rest_required_fields(request_type=document_processor_service.FetchProcessorTypesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_processor_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_processor_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_processor_service.FetchProcessorTypesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_processor_service.FetchProcessorTypesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_processor_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_processor_types_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_processor_types._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_processor_types_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_fetch_processor_types') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_fetch_processor_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.FetchProcessorTypesRequest.pb(document_processor_service.FetchProcessorTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_processor_service.FetchProcessorTypesResponse.to_json(document_processor_service.FetchProcessorTypesResponse())
        request = document_processor_service.FetchProcessorTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_processor_service.FetchProcessorTypesResponse()
        client.fetch_processor_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_processor_types_rest_bad_request(transport: str='rest', request_type=document_processor_service.FetchProcessorTypesRequest):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_processor_types(request)

def test_fetch_processor_types_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.FetchProcessorTypesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.FetchProcessorTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.fetch_processor_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}:fetchProcessorTypes' % client.transport._host, args[1])

def test_fetch_processor_types_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.fetch_processor_types(document_processor_service.FetchProcessorTypesRequest(), parent='parent_value')

def test_fetch_processor_types_rest_error():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.ListProcessorTypesRequest, dict])
def test_list_processor_types_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListProcessorTypesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListProcessorTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_processor_types(request)
    assert isinstance(response, pagers.ListProcessorTypesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_processor_types_rest_required_fields(request_type=document_processor_service.ListProcessorTypesRequest):
    if False:
        return 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_processor_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_processor_types._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_processor_service.ListProcessorTypesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_processor_service.ListProcessorTypesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_processor_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_processor_types_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_processor_types._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_processor_types_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_list_processor_types') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_list_processor_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.ListProcessorTypesRequest.pb(document_processor_service.ListProcessorTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_processor_service.ListProcessorTypesResponse.to_json(document_processor_service.ListProcessorTypesResponse())
        request = document_processor_service.ListProcessorTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_processor_service.ListProcessorTypesResponse()
        client.list_processor_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_processor_types_rest_bad_request(transport: str='rest', request_type=document_processor_service.ListProcessorTypesRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_processor_types(request)

def test_list_processor_types_rest_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListProcessorTypesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListProcessorTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_processor_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/processorTypes' % client.transport._host, args[1])

def test_list_processor_types_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_processor_types(document_processor_service.ListProcessorTypesRequest(), parent='parent_value')

def test_list_processor_types_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType(), processor_type.ProcessorType()], next_page_token='abc'), document_processor_service.ListProcessorTypesResponse(processor_types=[], next_page_token='def'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType()], next_page_token='ghi'), document_processor_service.ListProcessorTypesResponse(processor_types=[processor_type.ProcessorType(), processor_type.ProcessorType()]))
        response = response + response
        response = tuple((document_processor_service.ListProcessorTypesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_processor_types(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, processor_type.ProcessorType) for i in results))
        pages = list(client.list_processor_types(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.GetProcessorTypeRequest, dict])
def test_get_processor_type_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processorTypes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = processor_type.ProcessorType(name='name_value', type_='type__value', category='category_value', allow_creation=True, launch_stage=launch_stage_pb2.LaunchStage.UNIMPLEMENTED, sample_document_uris=['sample_document_uris_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = processor_type.ProcessorType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_processor_type(request)
    assert isinstance(response, processor_type.ProcessorType)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.category == 'category_value'
    assert response.allow_creation is True
    assert response.launch_stage == launch_stage_pb2.LaunchStage.UNIMPLEMENTED
    assert response.sample_document_uris == ['sample_document_uris_value']

def test_get_processor_type_rest_required_fields(request_type=document_processor_service.GetProcessorTypeRequest):
    if False:
        return 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_processor_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_processor_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = processor_type.ProcessorType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = processor_type.ProcessorType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_processor_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_processor_type_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_processor_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_processor_type_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_get_processor_type') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_get_processor_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.GetProcessorTypeRequest.pb(document_processor_service.GetProcessorTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = processor_type.ProcessorType.to_json(processor_type.ProcessorType())
        request = document_processor_service.GetProcessorTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = processor_type.ProcessorType()
        client.get_processor_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_processor_type_rest_bad_request(transport: str='rest', request_type=document_processor_service.GetProcessorTypeRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processorTypes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_processor_type(request)

def test_get_processor_type_rest_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = processor_type.ProcessorType()
        sample_request = {'name': 'projects/sample1/locations/sample2/processorTypes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = processor_type.ProcessorType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_processor_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processorTypes/*}' % client.transport._host, args[1])

def test_get_processor_type_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_processor_type(document_processor_service.GetProcessorTypeRequest(), name='name_value')

def test_get_processor_type_rest_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.ListProcessorsRequest, dict])
def test_list_processors_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListProcessorsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListProcessorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_processors(request)
    assert isinstance(response, pagers.ListProcessorsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_processors_rest_required_fields(request_type=document_processor_service.ListProcessorsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_processors._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_processors._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_processor_service.ListProcessorsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_processor_service.ListProcessorsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_processors(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_processors_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_processors._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_processors_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_list_processors') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_list_processors') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.ListProcessorsRequest.pb(document_processor_service.ListProcessorsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_processor_service.ListProcessorsResponse.to_json(document_processor_service.ListProcessorsResponse())
        request = document_processor_service.ListProcessorsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_processor_service.ListProcessorsResponse()
        client.list_processors(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_processors_rest_bad_request(transport: str='rest', request_type=document_processor_service.ListProcessorsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_processors(request)

def test_list_processors_rest_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListProcessorsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListProcessorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_processors(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/processors' % client.transport._host, args[1])

def test_list_processors_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_processors(document_processor_service.ListProcessorsRequest(), parent='parent_value')

def test_list_processors_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor(), processor.Processor()], next_page_token='abc'), document_processor_service.ListProcessorsResponse(processors=[], next_page_token='def'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor()], next_page_token='ghi'), document_processor_service.ListProcessorsResponse(processors=[processor.Processor(), processor.Processor()]))
        response = response + response
        response = tuple((document_processor_service.ListProcessorsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_processors(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, processor.Processor) for i in results))
        pages = list(client.list_processors(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.GetProcessorRequest, dict])
def test_get_processor_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = processor.Processor(name='name_value', type_='type__value', display_name='display_name_value', state=processor.Processor.State.ENABLED, default_processor_version='default_processor_version_value', process_endpoint='process_endpoint_value', kms_key_name='kms_key_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = processor.Processor.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_processor(request)
    assert isinstance(response, processor.Processor)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.display_name == 'display_name_value'
    assert response.state == processor.Processor.State.ENABLED
    assert response.default_processor_version == 'default_processor_version_value'
    assert response.process_endpoint == 'process_endpoint_value'
    assert response.kms_key_name == 'kms_key_name_value'

def test_get_processor_rest_required_fields(request_type=document_processor_service.GetProcessorRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = processor.Processor()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = processor.Processor.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_processor(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_processor_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_processor._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_processor_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_get_processor') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_get_processor') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.GetProcessorRequest.pb(document_processor_service.GetProcessorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = processor.Processor.to_json(processor.Processor())
        request = document_processor_service.GetProcessorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = processor.Processor()
        client.get_processor(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_processor_rest_bad_request(transport: str='rest', request_type=document_processor_service.GetProcessorRequest):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_processor(request)

def test_get_processor_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = processor.Processor()
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = processor.Processor.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_processor(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*}' % client.transport._host, args[1])

def test_get_processor_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_processor(document_processor_service.GetProcessorRequest(), name='name_value')

def test_get_processor_rest_error():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.TrainProcessorVersionRequest, dict])
def test_train_processor_version_rest(request_type):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.train_processor_version(request)
    assert response.operation.name == 'operations/spam'

def test_train_processor_version_rest_required_fields(request_type=document_processor_service.TrainProcessorVersionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).train_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).train_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.train_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_train_processor_version_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.train_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'processorVersion'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_train_processor_version_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_train_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_train_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.TrainProcessorVersionRequest.pb(document_processor_service.TrainProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.TrainProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.train_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_train_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.TrainProcessorVersionRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.train_processor_version(request)

def test_train_processor_version_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
        mock_args = dict(parent='parent_value', processor_version=processor.ProcessorVersion(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.train_processor_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/processors/*}/processorVersions:train' % client.transport._host, args[1])

def test_train_processor_version_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.train_processor_version(document_processor_service.TrainProcessorVersionRequest(), parent='parent_value', processor_version=processor.ProcessorVersion(name='name_value'))

def test_train_processor_version_rest_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.GetProcessorVersionRequest, dict])
def test_get_processor_version_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = processor.ProcessorVersion(name='name_value', display_name='display_name_value', state=processor.ProcessorVersion.State.DEPLOYED, kms_key_name='kms_key_name_value', kms_key_version_name='kms_key_version_name_value', google_managed=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = processor.ProcessorVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_processor_version(request)
    assert isinstance(response, processor.ProcessorVersion)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == processor.ProcessorVersion.State.DEPLOYED
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.kms_key_version_name == 'kms_key_version_name_value'
    assert response.google_managed is True

def test_get_processor_version_rest_required_fields(request_type=document_processor_service.GetProcessorVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = processor.ProcessorVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = processor.ProcessorVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_processor_version_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_processor_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_get_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_get_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.GetProcessorVersionRequest.pb(document_processor_service.GetProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = processor.ProcessorVersion.to_json(processor.ProcessorVersion())
        request = document_processor_service.GetProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = processor.ProcessorVersion()
        client.get_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.GetProcessorVersionRequest):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_processor_version(request)

def test_get_processor_version_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = processor.ProcessorVersion()
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = processor.ProcessorVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_processor_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*/processorVersions/*}' % client.transport._host, args[1])

def test_get_processor_version_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_processor_version(document_processor_service.GetProcessorVersionRequest(), name='name_value')

def test_get_processor_version_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.ListProcessorVersionsRequest, dict])
def test_list_processor_versions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListProcessorVersionsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListProcessorVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_processor_versions(request)
    assert isinstance(response, pagers.ListProcessorVersionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_processor_versions_rest_required_fields(request_type=document_processor_service.ListProcessorVersionsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_processor_versions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_processor_versions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_processor_service.ListProcessorVersionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_processor_service.ListProcessorVersionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_processor_versions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_processor_versions_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_processor_versions._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_processor_versions_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_list_processor_versions') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_list_processor_versions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.ListProcessorVersionsRequest.pb(document_processor_service.ListProcessorVersionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_processor_service.ListProcessorVersionsResponse.to_json(document_processor_service.ListProcessorVersionsResponse())
        request = document_processor_service.ListProcessorVersionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_processor_service.ListProcessorVersionsResponse()
        client.list_processor_versions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_processor_versions_rest_bad_request(transport: str='rest', request_type=document_processor_service.ListProcessorVersionsRequest):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_processor_versions(request)

def test_list_processor_versions_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListProcessorVersionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListProcessorVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_processor_versions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/processors/*}/processorVersions' % client.transport._host, args[1])

def test_list_processor_versions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_processor_versions(document_processor_service.ListProcessorVersionsRequest(), parent='parent_value')

def test_list_processor_versions_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion(), processor.ProcessorVersion()], next_page_token='abc'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[], next_page_token='def'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion()], next_page_token='ghi'), document_processor_service.ListProcessorVersionsResponse(processor_versions=[processor.ProcessorVersion(), processor.ProcessorVersion()]))
        response = response + response
        response = tuple((document_processor_service.ListProcessorVersionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/processors/sample3'}
        pager = client.list_processor_versions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, processor.ProcessorVersion) for i in results))
        pages = list(client.list_processor_versions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [document_processor_service.DeleteProcessorVersionRequest, dict])
def test_delete_processor_version_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_processor_version(request)
    assert response.operation.name == 'operations/spam'

def test_delete_processor_version_rest_required_fields(request_type=document_processor_service.DeleteProcessorVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_processor_version_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_processor_version_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_delete_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_delete_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.DeleteProcessorVersionRequest.pb(document_processor_service.DeleteProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.DeleteProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.DeleteProcessorVersionRequest):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_processor_version(request)

def test_delete_processor_version_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_processor_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*/processorVersions/*}' % client.transport._host, args[1])

def test_delete_processor_version_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_processor_version(document_processor_service.DeleteProcessorVersionRequest(), name='name_value')

def test_delete_processor_version_rest_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.DeployProcessorVersionRequest, dict])
def test_deploy_processor_version_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.deploy_processor_version(request)
    assert response.operation.name == 'operations/spam'

def test_deploy_processor_version_rest_required_fields(request_type=document_processor_service.DeployProcessorVersionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).deploy_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).deploy_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.deploy_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_deploy_processor_version_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.deploy_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_deploy_processor_version_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_deploy_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_deploy_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.DeployProcessorVersionRequest.pb(document_processor_service.DeployProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.DeployProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.deploy_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_deploy_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.DeployProcessorVersionRequest):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.deploy_processor_version(request)

def test_deploy_processor_version_rest_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.deploy_processor_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*/processorVersions/*}:deploy' % client.transport._host, args[1])

def test_deploy_processor_version_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.deploy_processor_version(document_processor_service.DeployProcessorVersionRequest(), name='name_value')

def test_deploy_processor_version_rest_error():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.UndeployProcessorVersionRequest, dict])
def test_undeploy_processor_version_rest(request_type):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.undeploy_processor_version(request)
    assert response.operation.name == 'operations/spam'

def test_undeploy_processor_version_rest_required_fields(request_type=document_processor_service.UndeployProcessorVersionRequest):
    if False:
        return 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undeploy_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undeploy_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.undeploy_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_undeploy_processor_version_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.undeploy_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_undeploy_processor_version_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_undeploy_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_undeploy_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.UndeployProcessorVersionRequest.pb(document_processor_service.UndeployProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.UndeployProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.undeploy_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_undeploy_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.UndeployProcessorVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.undeploy_processor_version(request)

def test_undeploy_processor_version_rest_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.undeploy_processor_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*/processorVersions/*}:undeploy' % client.transport._host, args[1])

def test_undeploy_processor_version_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.undeploy_processor_version(document_processor_service.UndeployProcessorVersionRequest(), name='name_value')

def test_undeploy_processor_version_rest_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.CreateProcessorRequest, dict])
def test_create_processor_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['processor'] = {'name': 'name_value', 'type_': 'type__value', 'display_name': 'display_name_value', 'state': 1, 'default_processor_version': 'default_processor_version_value', 'processor_version_aliases': [{'alias': 'alias_value', 'processor_version': 'processor_version_value'}], 'process_endpoint': 'process_endpoint_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'kms_key_name': 'kms_key_name_value'}
    test_field = document_processor_service.CreateProcessorRequest.meta.fields['processor']

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
    for (field, value) in request_init['processor'].items():
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
                for i in range(0, len(request_init['processor'][field])):
                    del request_init['processor'][field][i][subfield]
            else:
                del request_init['processor'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_processor.Processor(name='name_value', type_='type__value', display_name='display_name_value', state=gcd_processor.Processor.State.ENABLED, default_processor_version='default_processor_version_value', process_endpoint='process_endpoint_value', kms_key_name='kms_key_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_processor.Processor.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_processor(request)
    assert isinstance(response, gcd_processor.Processor)
    assert response.name == 'name_value'
    assert response.type_ == 'type__value'
    assert response.display_name == 'display_name_value'
    assert response.state == gcd_processor.Processor.State.ENABLED
    assert response.default_processor_version == 'default_processor_version_value'
    assert response.process_endpoint == 'process_endpoint_value'
    assert response.kms_key_name == 'kms_key_name_value'

def test_create_processor_rest_required_fields(request_type=document_processor_service.CreateProcessorRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_processor.Processor()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_processor.Processor.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_processor(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_processor_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_processor._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'processor'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_processor_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_create_processor') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_create_processor') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.CreateProcessorRequest.pb(document_processor_service.CreateProcessorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_processor.Processor.to_json(gcd_processor.Processor())
        request = document_processor_service.CreateProcessorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_processor.Processor()
        client.create_processor(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_processor_rest_bad_request(transport: str='rest', request_type=document_processor_service.CreateProcessorRequest):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_processor(request)

def test_create_processor_rest_flattened():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_processor.Processor()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', processor=gcd_processor.Processor(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_processor.Processor.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_processor(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/processors' % client.transport._host, args[1])

def test_create_processor_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_processor(document_processor_service.CreateProcessorRequest(), parent='parent_value', processor=gcd_processor.Processor(name='name_value'))

def test_create_processor_rest_error():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.DeleteProcessorRequest, dict])
def test_delete_processor_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_processor(request)
    assert response.operation.name == 'operations/spam'

def test_delete_processor_rest_required_fields(request_type=document_processor_service.DeleteProcessorRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_processor(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_processor_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_processor._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_processor_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_delete_processor') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_delete_processor') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.DeleteProcessorRequest.pb(document_processor_service.DeleteProcessorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.DeleteProcessorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_processor(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_processor_rest_bad_request(transport: str='rest', request_type=document_processor_service.DeleteProcessorRequest):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_processor(request)

def test_delete_processor_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_processor(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*}' % client.transport._host, args[1])

def test_delete_processor_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_processor(document_processor_service.DeleteProcessorRequest(), name='name_value')

def test_delete_processor_rest_error():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.EnableProcessorRequest, dict])
def test_enable_processor_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.enable_processor(request)
    assert response.operation.name == 'operations/spam'

def test_enable_processor_rest_required_fields(request_type=document_processor_service.EnableProcessorRequest):
    if False:
        return 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.enable_processor(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_enable_processor_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.enable_processor._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_enable_processor_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_enable_processor') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_enable_processor') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.EnableProcessorRequest.pb(document_processor_service.EnableProcessorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.EnableProcessorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.enable_processor(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_enable_processor_rest_bad_request(transport: str='rest', request_type=document_processor_service.EnableProcessorRequest):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.enable_processor(request)

def test_enable_processor_rest_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.DisableProcessorRequest, dict])
def test_disable_processor_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.disable_processor(request)
    assert response.operation.name == 'operations/spam'

def test_disable_processor_rest_required_fields(request_type=document_processor_service.DisableProcessorRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_processor._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.disable_processor(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_disable_processor_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.disable_processor._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_disable_processor_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_disable_processor') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_disable_processor') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.DisableProcessorRequest.pb(document_processor_service.DisableProcessorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.DisableProcessorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.disable_processor(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_disable_processor_rest_bad_request(transport: str='rest', request_type=document_processor_service.DisableProcessorRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.disable_processor(request)

def test_disable_processor_rest_error():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.SetDefaultProcessorVersionRequest, dict])
def test_set_default_processor_version_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'processor': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_default_processor_version(request)
    assert response.operation.name == 'operations/spam'

def test_set_default_processor_version_rest_required_fields(request_type=document_processor_service.SetDefaultProcessorVersionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['processor'] = ''
    request_init['default_processor_version'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_default_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['processor'] = 'processor_value'
    jsonified_request['defaultProcessorVersion'] = 'default_processor_version_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_default_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'processor' in jsonified_request
    assert jsonified_request['processor'] == 'processor_value'
    assert 'defaultProcessorVersion' in jsonified_request
    assert jsonified_request['defaultProcessorVersion'] == 'default_processor_version_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_default_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_default_processor_version_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_default_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('processor', 'defaultProcessorVersion'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_default_processor_version_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_set_default_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_set_default_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.SetDefaultProcessorVersionRequest.pb(document_processor_service.SetDefaultProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.SetDefaultProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.set_default_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_default_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.SetDefaultProcessorVersionRequest):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'processor': 'projects/sample1/locations/sample2/processors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_default_processor_version(request)

def test_set_default_processor_version_rest_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.ReviewDocumentRequest, dict])
def test_review_document_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'human_review_config': 'projects/sample1/locations/sample2/processors/sample3/humanReviewConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.review_document(request)
    assert response.operation.name == 'operations/spam'

def test_review_document_rest_required_fields(request_type=document_processor_service.ReviewDocumentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['human_review_config'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).review_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['humanReviewConfig'] = 'human_review_config_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).review_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'humanReviewConfig' in jsonified_request
    assert jsonified_request['humanReviewConfig'] == 'human_review_config_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.review_document(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_review_document_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.review_document._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('humanReviewConfig',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_review_document_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_review_document') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_review_document') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.ReviewDocumentRequest.pb(document_processor_service.ReviewDocumentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.ReviewDocumentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.review_document(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_review_document_rest_bad_request(transport: str='rest', request_type=document_processor_service.ReviewDocumentRequest):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'human_review_config': 'projects/sample1/locations/sample2/processors/sample3/humanReviewConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.review_document(request)

def test_review_document_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'human_review_config': 'projects/sample1/locations/sample2/processors/sample3/humanReviewConfig'}
        mock_args = dict(human_review_config='human_review_config_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.review_document(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{human_review_config=projects/*/locations/*/processors/*/humanReviewConfig}:reviewDocument' % client.transport._host, args[1])

def test_review_document_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.review_document(document_processor_service.ReviewDocumentRequest(), human_review_config='human_review_config_value')

def test_review_document_rest_error():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.EvaluateProcessorVersionRequest, dict])
def test_evaluate_processor_version_rest(request_type):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'processor_version': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.evaluate_processor_version(request)
    assert response.operation.name == 'operations/spam'

def test_evaluate_processor_version_rest_required_fields(request_type=document_processor_service.EvaluateProcessorVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['processor_version'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).evaluate_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['processorVersion'] = 'processor_version_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).evaluate_processor_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'processorVersion' in jsonified_request
    assert jsonified_request['processorVersion'] == 'processor_version_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.evaluate_processor_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_evaluate_processor_version_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.evaluate_processor_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('processorVersion',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_evaluate_processor_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_evaluate_processor_version') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_evaluate_processor_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.EvaluateProcessorVersionRequest.pb(document_processor_service.EvaluateProcessorVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = document_processor_service.EvaluateProcessorVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.evaluate_processor_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_evaluate_processor_version_rest_bad_request(transport: str='rest', request_type=document_processor_service.EvaluateProcessorVersionRequest):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'processor_version': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.evaluate_processor_version(request)

def test_evaluate_processor_version_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'processor_version': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        mock_args = dict(processor_version='processor_version_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.evaluate_processor_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{processor_version=projects/*/locations/*/processors/*/processorVersions/*}:evaluateProcessorVersion' % client.transport._host, args[1])

def test_evaluate_processor_version_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.evaluate_processor_version(document_processor_service.EvaluateProcessorVersionRequest(), processor_version='processor_version_value')

def test_evaluate_processor_version_rest_error():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.GetEvaluationRequest, dict])
def test_get_evaluation_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4/evaluations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = evaluation.Evaluation(name='name_value', kms_key_name='kms_key_name_value', kms_key_version_name='kms_key_version_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = evaluation.Evaluation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_evaluation(request)
    assert isinstance(response, evaluation.Evaluation)
    assert response.name == 'name_value'
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.kms_key_version_name == 'kms_key_version_name_value'

def test_get_evaluation_rest_required_fields(request_type=document_processor_service.GetEvaluationRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_evaluation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_evaluation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = evaluation.Evaluation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = evaluation.Evaluation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_evaluation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_evaluation_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_evaluation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_evaluation_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_get_evaluation') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_get_evaluation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.GetEvaluationRequest.pb(document_processor_service.GetEvaluationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = evaluation.Evaluation.to_json(evaluation.Evaluation())
        request = document_processor_service.GetEvaluationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = evaluation.Evaluation()
        client.get_evaluation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_evaluation_rest_bad_request(transport: str='rest', request_type=document_processor_service.GetEvaluationRequest):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4/evaluations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_evaluation(request)

def test_get_evaluation_rest_flattened():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = evaluation.Evaluation()
        sample_request = {'name': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4/evaluations/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = evaluation.Evaluation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_evaluation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/processors/*/processorVersions/*/evaluations/*}' % client.transport._host, args[1])

def test_get_evaluation_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_evaluation(document_processor_service.GetEvaluationRequest(), name='name_value')

def test_get_evaluation_rest_error():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [document_processor_service.ListEvaluationsRequest, dict])
def test_list_evaluations_rest(request_type):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListEvaluationsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListEvaluationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_evaluations(request)
    assert isinstance(response, pagers.ListEvaluationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_evaluations_rest_required_fields(request_type=document_processor_service.ListEvaluationsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DocumentProcessorServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_evaluations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_evaluations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = document_processor_service.ListEvaluationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = document_processor_service.ListEvaluationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_evaluations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_evaluations_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_evaluations._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_evaluations_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DocumentProcessorServiceRestInterceptor())
    client = DocumentProcessorServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'post_list_evaluations') as post, mock.patch.object(transports.DocumentProcessorServiceRestInterceptor, 'pre_list_evaluations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = document_processor_service.ListEvaluationsRequest.pb(document_processor_service.ListEvaluationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = document_processor_service.ListEvaluationsResponse.to_json(document_processor_service.ListEvaluationsResponse())
        request = document_processor_service.ListEvaluationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = document_processor_service.ListEvaluationsResponse()
        client.list_evaluations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_evaluations_rest_bad_request(transport: str='rest', request_type=document_processor_service.ListEvaluationsRequest):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_evaluations(request)

def test_list_evaluations_rest_flattened():
    if False:
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = document_processor_service.ListEvaluationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = document_processor_service.ListEvaluationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_evaluations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/processors/*/processorVersions/*}/evaluations' % client.transport._host, args[1])

def test_list_evaluations_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_evaluations(document_processor_service.ListEvaluationsRequest(), parent='parent_value')

def test_list_evaluations_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation(), evaluation.Evaluation()], next_page_token='abc'), document_processor_service.ListEvaluationsResponse(evaluations=[], next_page_token='def'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation()], next_page_token='ghi'), document_processor_service.ListEvaluationsResponse(evaluations=[evaluation.Evaluation(), evaluation.Evaluation()]))
        response = response + response
        response = tuple((document_processor_service.ListEvaluationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/processors/sample3/processorVersions/sample4'}
        pager = client.list_evaluations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, evaluation.Evaluation) for i in results))
        pages = list(client.list_evaluations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.DocumentProcessorServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DocumentProcessorServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DocumentProcessorServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DocumentProcessorServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DocumentProcessorServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DocumentProcessorServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DocumentProcessorServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DocumentProcessorServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DocumentProcessorServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.DocumentProcessorServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DocumentProcessorServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceGrpcAsyncIOTransport, transports.DocumentProcessorServiceRestTransport])
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
        print('Hello World!')
    transport = DocumentProcessorServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DocumentProcessorServiceGrpcTransport)

def test_document_processor_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DocumentProcessorServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_document_processor_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.documentai_v1.services.document_processor_service.transports.DocumentProcessorServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DocumentProcessorServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('process_document', 'batch_process_documents', 'fetch_processor_types', 'list_processor_types', 'get_processor_type', 'list_processors', 'get_processor', 'train_processor_version', 'get_processor_version', 'list_processor_versions', 'delete_processor_version', 'deploy_processor_version', 'undeploy_processor_version', 'create_processor', 'delete_processor', 'enable_processor', 'disable_processor', 'set_default_processor_version', 'review_document', 'evaluate_processor_version', 'get_evaluation', 'list_evaluations', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_document_processor_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.documentai_v1.services.document_processor_service.transports.DocumentProcessorServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DocumentProcessorServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_document_processor_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.documentai_v1.services.document_processor_service.transports.DocumentProcessorServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DocumentProcessorServiceTransport()
        adc.assert_called_once()

def test_document_processor_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DocumentProcessorServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceGrpcAsyncIOTransport])
def test_document_processor_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceGrpcAsyncIOTransport, transports.DocumentProcessorServiceRestTransport])
def test_document_processor_service_transport_auth_gdch_credentials(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DocumentProcessorServiceGrpcTransport, grpc_helpers), (transports.DocumentProcessorServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_document_processor_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('documentai.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='documentai.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceGrpcAsyncIOTransport])
def test_document_processor_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_document_processor_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.DocumentProcessorServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_document_processor_service_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_document_processor_service_host_no_port(transport_name):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='documentai.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('documentai.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_document_processor_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='documentai.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('documentai.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://documentai.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_document_processor_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = DocumentProcessorServiceClient(credentials=creds1, transport=transport_name)
    client2 = DocumentProcessorServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.process_document._session
    session2 = client2.transport.process_document._session
    assert session1 != session2
    session1 = client1.transport.batch_process_documents._session
    session2 = client2.transport.batch_process_documents._session
    assert session1 != session2
    session1 = client1.transport.fetch_processor_types._session
    session2 = client2.transport.fetch_processor_types._session
    assert session1 != session2
    session1 = client1.transport.list_processor_types._session
    session2 = client2.transport.list_processor_types._session
    assert session1 != session2
    session1 = client1.transport.get_processor_type._session
    session2 = client2.transport.get_processor_type._session
    assert session1 != session2
    session1 = client1.transport.list_processors._session
    session2 = client2.transport.list_processors._session
    assert session1 != session2
    session1 = client1.transport.get_processor._session
    session2 = client2.transport.get_processor._session
    assert session1 != session2
    session1 = client1.transport.train_processor_version._session
    session2 = client2.transport.train_processor_version._session
    assert session1 != session2
    session1 = client1.transport.get_processor_version._session
    session2 = client2.transport.get_processor_version._session
    assert session1 != session2
    session1 = client1.transport.list_processor_versions._session
    session2 = client2.transport.list_processor_versions._session
    assert session1 != session2
    session1 = client1.transport.delete_processor_version._session
    session2 = client2.transport.delete_processor_version._session
    assert session1 != session2
    session1 = client1.transport.deploy_processor_version._session
    session2 = client2.transport.deploy_processor_version._session
    assert session1 != session2
    session1 = client1.transport.undeploy_processor_version._session
    session2 = client2.transport.undeploy_processor_version._session
    assert session1 != session2
    session1 = client1.transport.create_processor._session
    session2 = client2.transport.create_processor._session
    assert session1 != session2
    session1 = client1.transport.delete_processor._session
    session2 = client2.transport.delete_processor._session
    assert session1 != session2
    session1 = client1.transport.enable_processor._session
    session2 = client2.transport.enable_processor._session
    assert session1 != session2
    session1 = client1.transport.disable_processor._session
    session2 = client2.transport.disable_processor._session
    assert session1 != session2
    session1 = client1.transport.set_default_processor_version._session
    session2 = client2.transport.set_default_processor_version._session
    assert session1 != session2
    session1 = client1.transport.review_document._session
    session2 = client2.transport.review_document._session
    assert session1 != session2
    session1 = client1.transport.evaluate_processor_version._session
    session2 = client2.transport.evaluate_processor_version._session
    assert session1 != session2
    session1 = client1.transport.get_evaluation._session
    session2 = client2.transport.get_evaluation._session
    assert session1 != session2
    session1 = client1.transport.list_evaluations._session
    session2 = client2.transport.list_evaluations._session
    assert session1 != session2

def test_document_processor_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DocumentProcessorServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_document_processor_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DocumentProcessorServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceGrpcAsyncIOTransport])
def test_document_processor_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('transport_class', [transports.DocumentProcessorServiceGrpcTransport, transports.DocumentProcessorServiceGrpcAsyncIOTransport])
def test_document_processor_service_transport_channel_mtls_with_adc(transport_class):
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

def test_document_processor_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_document_processor_service_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_evaluation_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    processor = 'whelk'
    processor_version = 'octopus'
    evaluation = 'oyster'
    expected = 'projects/{project}/locations/{location}/processors/{processor}/processorVersions/{processor_version}/evaluations/{evaluation}'.format(project=project, location=location, processor=processor, processor_version=processor_version, evaluation=evaluation)
    actual = DocumentProcessorServiceClient.evaluation_path(project, location, processor, processor_version, evaluation)
    assert expected == actual

def test_parse_evaluation_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'processor': 'mussel', 'processor_version': 'winkle', 'evaluation': 'nautilus'}
    path = DocumentProcessorServiceClient.evaluation_path(**expected)
    actual = DocumentProcessorServiceClient.parse_evaluation_path(path)
    assert expected == actual

def test_human_review_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    location = 'abalone'
    processor = 'squid'
    expected = 'projects/{project}/locations/{location}/processors/{processor}/humanReviewConfig'.format(project=project, location=location, processor=processor)
    actual = DocumentProcessorServiceClient.human_review_config_path(project, location, processor)
    assert expected == actual

def test_parse_human_review_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam', 'location': 'whelk', 'processor': 'octopus'}
    path = DocumentProcessorServiceClient.human_review_config_path(**expected)
    actual = DocumentProcessorServiceClient.parse_human_review_config_path(path)
    assert expected == actual

def test_processor_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    processor = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/processors/{processor}'.format(project=project, location=location, processor=processor)
    actual = DocumentProcessorServiceClient.processor_path(project, location, processor)
    assert expected == actual

def test_parse_processor_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'location': 'winkle', 'processor': 'nautilus'}
    path = DocumentProcessorServiceClient.processor_path(**expected)
    actual = DocumentProcessorServiceClient.parse_processor_path(path)
    assert expected == actual

def test_processor_type_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    location = 'abalone'
    processor_type = 'squid'
    expected = 'projects/{project}/locations/{location}/processorTypes/{processor_type}'.format(project=project, location=location, processor_type=processor_type)
    actual = DocumentProcessorServiceClient.processor_type_path(project, location, processor_type)
    assert expected == actual

def test_parse_processor_type_path():
    if False:
        return 10
    expected = {'project': 'clam', 'location': 'whelk', 'processor_type': 'octopus'}
    path = DocumentProcessorServiceClient.processor_type_path(**expected)
    actual = DocumentProcessorServiceClient.parse_processor_type_path(path)
    assert expected == actual

def test_processor_version_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    processor = 'cuttlefish'
    processor_version = 'mussel'
    expected = 'projects/{project}/locations/{location}/processors/{processor}/processorVersions/{processor_version}'.format(project=project, location=location, processor=processor, processor_version=processor_version)
    actual = DocumentProcessorServiceClient.processor_version_path(project, location, processor, processor_version)
    assert expected == actual

def test_parse_processor_version_path():
    if False:
        print('Hello World!')
    expected = {'project': 'winkle', 'location': 'nautilus', 'processor': 'scallop', 'processor_version': 'abalone'}
    path = DocumentProcessorServiceClient.processor_version_path(**expected)
    actual = DocumentProcessorServiceClient.parse_processor_version_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DocumentProcessorServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'clam'}
    path = DocumentProcessorServiceClient.common_billing_account_path(**expected)
    actual = DocumentProcessorServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DocumentProcessorServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = DocumentProcessorServiceClient.common_folder_path(**expected)
    actual = DocumentProcessorServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DocumentProcessorServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nudibranch'}
    path = DocumentProcessorServiceClient.common_organization_path(**expected)
    actual = DocumentProcessorServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = DocumentProcessorServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = DocumentProcessorServiceClient.common_project_path(**expected)
    actual = DocumentProcessorServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DocumentProcessorServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = DocumentProcessorServiceClient.common_location_path(**expected)
    actual = DocumentProcessorServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DocumentProcessorServiceTransport, '_prep_wrapped_messages') as prep:
        client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DocumentProcessorServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DocumentProcessorServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        return 10
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = DocumentProcessorServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = DocumentProcessorServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DocumentProcessorServiceClient, transports.DocumentProcessorServiceGrpcTransport), (DocumentProcessorServiceAsyncClient, transports.DocumentProcessorServiceGrpcAsyncIOTransport)])
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