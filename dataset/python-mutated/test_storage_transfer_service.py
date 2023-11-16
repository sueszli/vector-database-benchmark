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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import date_pb2
from google.type import timeofday_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.storage_transfer_v1.services.storage_transfer_service import StorageTransferServiceAsyncClient, StorageTransferServiceClient, pagers, transports
from google.cloud.storage_transfer_v1.types import transfer, transfer_types

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
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert StorageTransferServiceClient._get_default_mtls_endpoint(None) is None
    assert StorageTransferServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert StorageTransferServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert StorageTransferServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert StorageTransferServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert StorageTransferServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(StorageTransferServiceClient, 'grpc'), (StorageTransferServiceAsyncClient, 'grpc_asyncio'), (StorageTransferServiceClient, 'rest')])
def test_storage_transfer_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('storagetransfer.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://storagetransfer.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.StorageTransferServiceGrpcTransport, 'grpc'), (transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.StorageTransferServiceRestTransport, 'rest')])
def test_storage_transfer_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(StorageTransferServiceClient, 'grpc'), (StorageTransferServiceAsyncClient, 'grpc_asyncio'), (StorageTransferServiceClient, 'rest')])
def test_storage_transfer_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('storagetransfer.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://storagetransfer.googleapis.com')

def test_storage_transfer_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = StorageTransferServiceClient.get_transport_class()
    available_transports = [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceRestTransport]
    assert transport in available_transports
    transport = StorageTransferServiceClient.get_transport_class('grpc')
    assert transport == transports.StorageTransferServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport, 'grpc'), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (StorageTransferServiceClient, transports.StorageTransferServiceRestTransport, 'rest')])
@mock.patch.object(StorageTransferServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(StorageTransferServiceClient))
@mock.patch.object(StorageTransferServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(StorageTransferServiceAsyncClient))
def test_storage_transfer_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(StorageTransferServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(StorageTransferServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport, 'grpc', 'true'), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport, 'grpc', 'false'), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (StorageTransferServiceClient, transports.StorageTransferServiceRestTransport, 'rest', 'true'), (StorageTransferServiceClient, transports.StorageTransferServiceRestTransport, 'rest', 'false')])
@mock.patch.object(StorageTransferServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(StorageTransferServiceClient))
@mock.patch.object(StorageTransferServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(StorageTransferServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_storage_transfer_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [StorageTransferServiceClient, StorageTransferServiceAsyncClient])
@mock.patch.object(StorageTransferServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(StorageTransferServiceClient))
@mock.patch.object(StorageTransferServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(StorageTransferServiceAsyncClient))
def test_storage_transfer_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport, 'grpc'), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (StorageTransferServiceClient, transports.StorageTransferServiceRestTransport, 'rest')])
def test_storage_transfer_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport, 'grpc', grpc_helpers), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (StorageTransferServiceClient, transports.StorageTransferServiceRestTransport, 'rest', None)])
def test_storage_transfer_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_storage_transfer_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.storage_transfer_v1.services.storage_transfer_service.transports.StorageTransferServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = StorageTransferServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport, 'grpc', grpc_helpers), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_storage_transfer_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('storagetransfer.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='storagetransfer.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [transfer.GetGoogleServiceAccountRequest, dict])
def test_get_google_service_account(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_google_service_account), '__call__') as call:
        call.return_value = transfer_types.GoogleServiceAccount(account_email='account_email_value', subject_id='subject_id_value')
        response = client.get_google_service_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetGoogleServiceAccountRequest()
    assert isinstance(response, transfer_types.GoogleServiceAccount)
    assert response.account_email == 'account_email_value'
    assert response.subject_id == 'subject_id_value'

def test_get_google_service_account_empty_call():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_google_service_account), '__call__') as call:
        client.get_google_service_account()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetGoogleServiceAccountRequest()

@pytest.mark.asyncio
async def test_get_google_service_account_async(transport: str='grpc_asyncio', request_type=transfer.GetGoogleServiceAccountRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_google_service_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.GoogleServiceAccount(account_email='account_email_value', subject_id='subject_id_value'))
        response = await client.get_google_service_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetGoogleServiceAccountRequest()
    assert isinstance(response, transfer_types.GoogleServiceAccount)
    assert response.account_email == 'account_email_value'
    assert response.subject_id == 'subject_id_value'

@pytest.mark.asyncio
async def test_get_google_service_account_async_from_dict():
    await test_get_google_service_account_async(request_type=dict)

def test_get_google_service_account_field_headers():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.GetGoogleServiceAccountRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.get_google_service_account), '__call__') as call:
        call.return_value = transfer_types.GoogleServiceAccount()
        client.get_google_service_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_google_service_account_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.GetGoogleServiceAccountRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.get_google_service_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.GoogleServiceAccount())
        await client.get_google_service_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.CreateTransferJobRequest, dict])
def test_create_transfer_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_transfer_job), '__call__') as call:
        call.return_value = transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value')
        response = client.create_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.CreateTransferJobRequest()
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

def test_create_transfer_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_transfer_job), '__call__') as call:
        client.create_transfer_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.CreateTransferJobRequest()

@pytest.mark.asyncio
async def test_create_transfer_job_async(transport: str='grpc_asyncio', request_type=transfer.CreateTransferJobRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value'))
        response = await client.create_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.CreateTransferJobRequest()
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

@pytest.mark.asyncio
async def test_create_transfer_job_async_from_dict():
    await test_create_transfer_job_async(request_type=dict)

@pytest.mark.parametrize('request_type', [transfer.UpdateTransferJobRequest, dict])
def test_update_transfer_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_transfer_job), '__call__') as call:
        call.return_value = transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value')
        response = client.update_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.UpdateTransferJobRequest()
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

def test_update_transfer_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_transfer_job), '__call__') as call:
        client.update_transfer_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.UpdateTransferJobRequest()

@pytest.mark.asyncio
async def test_update_transfer_job_async(transport: str='grpc_asyncio', request_type=transfer.UpdateTransferJobRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value'))
        response = await client.update_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.UpdateTransferJobRequest()
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

@pytest.mark.asyncio
async def test_update_transfer_job_async_from_dict():
    await test_update_transfer_job_async(request_type=dict)

def test_update_transfer_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.UpdateTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.update_transfer_job), '__call__') as call:
        call.return_value = transfer_types.TransferJob()
        client.update_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_transfer_job_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.UpdateTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.update_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.TransferJob())
        await client.update_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.GetTransferJobRequest, dict])
def test_get_transfer_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transfer_job), '__call__') as call:
        call.return_value = transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value')
        response = client.get_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetTransferJobRequest()
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

def test_get_transfer_job_empty_call():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_transfer_job), '__call__') as call:
        client.get_transfer_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetTransferJobRequest()

@pytest.mark.asyncio
async def test_get_transfer_job_async(transport: str='grpc_asyncio', request_type=transfer.GetTransferJobRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value'))
        response = await client.get_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetTransferJobRequest()
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

@pytest.mark.asyncio
async def test_get_transfer_job_async_from_dict():
    await test_get_transfer_job_async(request_type=dict)

def test_get_transfer_job_field_headers():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.GetTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.get_transfer_job), '__call__') as call:
        call.return_value = transfer_types.TransferJob()
        client.get_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_transfer_job_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.GetTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.get_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.TransferJob())
        await client.get_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.ListTransferJobsRequest, dict])
def test_list_transfer_jobs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__') as call:
        call.return_value = transfer.ListTransferJobsResponse(next_page_token='next_page_token_value')
        response = client.list_transfer_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ListTransferJobsRequest()
    assert isinstance(response, pagers.ListTransferJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_jobs_empty_call():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__') as call:
        client.list_transfer_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ListTransferJobsRequest()

@pytest.mark.asyncio
async def test_list_transfer_jobs_async(transport: str='grpc_asyncio', request_type=transfer.ListTransferJobsRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.ListTransferJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_transfer_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ListTransferJobsRequest()
    assert isinstance(response, pagers.ListTransferJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transfer_jobs_async_from_dict():
    await test_list_transfer_jobs_async(request_type=dict)

def test_list_transfer_jobs_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__') as call:
        call.side_effect = (transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob(), transfer_types.TransferJob()], next_page_token='abc'), transfer.ListTransferJobsResponse(transfer_jobs=[], next_page_token='def'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob()], next_page_token='ghi'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob()]), RuntimeError)
        metadata = ()
        pager = client.list_transfer_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer_types.TransferJob) for i in results))

def test_list_transfer_jobs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__') as call:
        call.side_effect = (transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob(), transfer_types.TransferJob()], next_page_token='abc'), transfer.ListTransferJobsResponse(transfer_jobs=[], next_page_token='def'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob()], next_page_token='ghi'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob()]), RuntimeError)
        pages = list(client.list_transfer_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transfer_jobs_async_pager():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob(), transfer_types.TransferJob()], next_page_token='abc'), transfer.ListTransferJobsResponse(transfer_jobs=[], next_page_token='def'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob()], next_page_token='ghi'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob()]), RuntimeError)
        async_pager = await client.list_transfer_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, transfer_types.TransferJob) for i in responses))

@pytest.mark.asyncio
async def test_list_transfer_jobs_async_pages():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob(), transfer_types.TransferJob()], next_page_token='abc'), transfer.ListTransferJobsResponse(transfer_jobs=[], next_page_token='def'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob()], next_page_token='ghi'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transfer_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [transfer.PauseTransferOperationRequest, dict])
def test_pause_transfer_operation(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_transfer_operation), '__call__') as call:
        call.return_value = None
        response = client.pause_transfer_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.PauseTransferOperationRequest()
    assert response is None

def test_pause_transfer_operation_empty_call():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.pause_transfer_operation), '__call__') as call:
        client.pause_transfer_operation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.PauseTransferOperationRequest()

@pytest.mark.asyncio
async def test_pause_transfer_operation_async(transport: str='grpc_asyncio', request_type=transfer.PauseTransferOperationRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_transfer_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.pause_transfer_operation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.PauseTransferOperationRequest()
    assert response is None

@pytest.mark.asyncio
async def test_pause_transfer_operation_async_from_dict():
    await test_pause_transfer_operation_async(request_type=dict)

def test_pause_transfer_operation_field_headers():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.PauseTransferOperationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_transfer_operation), '__call__') as call:
        call.return_value = None
        client.pause_transfer_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_pause_transfer_operation_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.PauseTransferOperationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_transfer_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.pause_transfer_operation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.ResumeTransferOperationRequest, dict])
def test_resume_transfer_operation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_transfer_operation), '__call__') as call:
        call.return_value = None
        response = client.resume_transfer_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ResumeTransferOperationRequest()
    assert response is None

def test_resume_transfer_operation_empty_call():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_transfer_operation), '__call__') as call:
        client.resume_transfer_operation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ResumeTransferOperationRequest()

@pytest.mark.asyncio
async def test_resume_transfer_operation_async(transport: str='grpc_asyncio', request_type=transfer.ResumeTransferOperationRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_transfer_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.resume_transfer_operation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ResumeTransferOperationRequest()
    assert response is None

@pytest.mark.asyncio
async def test_resume_transfer_operation_async_from_dict():
    await test_resume_transfer_operation_async(request_type=dict)

def test_resume_transfer_operation_field_headers():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.ResumeTransferOperationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_transfer_operation), '__call__') as call:
        call.return_value = None
        client.resume_transfer_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_transfer_operation_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.ResumeTransferOperationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_transfer_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.resume_transfer_operation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.RunTransferJobRequest, dict])
def test_run_transfer_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_transfer_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.run_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.RunTransferJobRequest()
    assert isinstance(response, future.Future)

def test_run_transfer_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.run_transfer_job), '__call__') as call:
        client.run_transfer_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.RunTransferJobRequest()

@pytest.mark.asyncio
async def test_run_transfer_job_async(transport: str='grpc_asyncio', request_type=transfer.RunTransferJobRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.run_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.RunTransferJobRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_run_transfer_job_async_from_dict():
    await test_run_transfer_job_async(request_type=dict)

def test_run_transfer_job_field_headers():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.RunTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.run_transfer_job), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.run_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_run_transfer_job_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.RunTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.run_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.run_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.DeleteTransferJobRequest, dict])
def test_delete_transfer_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transfer_job), '__call__') as call:
        call.return_value = None
        response = client.delete_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.DeleteTransferJobRequest()
    assert response is None

def test_delete_transfer_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_transfer_job), '__call__') as call:
        client.delete_transfer_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.DeleteTransferJobRequest()

@pytest.mark.asyncio
async def test_delete_transfer_job_async(transport: str='grpc_asyncio', request_type=transfer.DeleteTransferJobRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.DeleteTransferJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_transfer_job_async_from_dict():
    await test_delete_transfer_job_async(request_type=dict)

def test_delete_transfer_job_field_headers():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.DeleteTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.delete_transfer_job), '__call__') as call:
        call.return_value = None
        client.delete_transfer_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_transfer_job_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.DeleteTransferJobRequest()
    request.job_name = 'job_name_value'
    with mock.patch.object(type(client.transport.delete_transfer_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_transfer_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job_name=job_name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [transfer.CreateAgentPoolRequest, dict])
def test_create_agent_pool(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING)
        response = client.create_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.CreateAgentPoolRequest()
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

def test_create_agent_pool_empty_call():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        client.create_agent_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.CreateAgentPoolRequest()

@pytest.mark.asyncio
async def test_create_agent_pool_async(transport: str='grpc_asyncio', request_type=transfer.CreateAgentPoolRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING))
        response = await client.create_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.CreateAgentPoolRequest()
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

@pytest.mark.asyncio
async def test_create_agent_pool_async_from_dict():
    await test_create_agent_pool_async(request_type=dict)

def test_create_agent_pool_field_headers():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.CreateAgentPoolRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        client.create_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_agent_pool_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.CreateAgentPoolRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool())
        await client.create_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

def test_create_agent_pool_flattened():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        client.create_agent_pool(project_id='project_id_value', agent_pool=transfer_types.AgentPool(name='name_value'), agent_pool_id='agent_pool_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].agent_pool
        mock_val = transfer_types.AgentPool(name='name_value')
        assert arg == mock_val
        arg = args[0].agent_pool_id
        mock_val = 'agent_pool_id_value'
        assert arg == mock_val

def test_create_agent_pool_flattened_error():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_agent_pool(transfer.CreateAgentPoolRequest(), project_id='project_id_value', agent_pool=transfer_types.AgentPool(name='name_value'), agent_pool_id='agent_pool_id_value')

@pytest.mark.asyncio
async def test_create_agent_pool_flattened_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool())
        response = await client.create_agent_pool(project_id='project_id_value', agent_pool=transfer_types.AgentPool(name='name_value'), agent_pool_id='agent_pool_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].agent_pool
        mock_val = transfer_types.AgentPool(name='name_value')
        assert arg == mock_val
        arg = args[0].agent_pool_id
        mock_val = 'agent_pool_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_agent_pool_flattened_error_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_agent_pool(transfer.CreateAgentPoolRequest(), project_id='project_id_value', agent_pool=transfer_types.AgentPool(name='name_value'), agent_pool_id='agent_pool_id_value')

@pytest.mark.parametrize('request_type', [transfer.UpdateAgentPoolRequest, dict])
def test_update_agent_pool(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING)
        response = client.update_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.UpdateAgentPoolRequest()
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

def test_update_agent_pool_empty_call():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        client.update_agent_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.UpdateAgentPoolRequest()

@pytest.mark.asyncio
async def test_update_agent_pool_async(transport: str='grpc_asyncio', request_type=transfer.UpdateAgentPoolRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING))
        response = await client.update_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.UpdateAgentPoolRequest()
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

@pytest.mark.asyncio
async def test_update_agent_pool_async_from_dict():
    await test_update_agent_pool_async(request_type=dict)

def test_update_agent_pool_field_headers():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.UpdateAgentPoolRequest()
    request.agent_pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        client.update_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'agent_pool.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_agent_pool_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.UpdateAgentPoolRequest()
    request.agent_pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool())
        await client.update_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'agent_pool.name=name_value') in kw['metadata']

def test_update_agent_pool_flattened():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        client.update_agent_pool(agent_pool=transfer_types.AgentPool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].agent_pool
        mock_val = transfer_types.AgentPool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_agent_pool_flattened_error():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_agent_pool(transfer.UpdateAgentPoolRequest(), agent_pool=transfer_types.AgentPool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_agent_pool_flattened_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool())
        response = await client.update_agent_pool(agent_pool=transfer_types.AgentPool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].agent_pool
        mock_val = transfer_types.AgentPool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_agent_pool_flattened_error_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_agent_pool(transfer.UpdateAgentPoolRequest(), agent_pool=transfer_types.AgentPool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [transfer.GetAgentPoolRequest, dict])
def test_get_agent_pool(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING)
        response = client.get_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetAgentPoolRequest()
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

def test_get_agent_pool_empty_call():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        client.get_agent_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetAgentPoolRequest()

@pytest.mark.asyncio
async def test_get_agent_pool_async(transport: str='grpc_asyncio', request_type=transfer.GetAgentPoolRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING))
        response = await client.get_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.GetAgentPoolRequest()
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

@pytest.mark.asyncio
async def test_get_agent_pool_async_from_dict():
    await test_get_agent_pool_async(request_type=dict)

def test_get_agent_pool_field_headers():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.GetAgentPoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        client.get_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_agent_pool_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.GetAgentPoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool())
        await client.get_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_agent_pool_flattened():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        client.get_agent_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_agent_pool_flattened_error():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_agent_pool(transfer.GetAgentPoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_agent_pool_flattened_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_agent_pool), '__call__') as call:
        call.return_value = transfer_types.AgentPool()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer_types.AgentPool())
        response = await client.get_agent_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_agent_pool_flattened_error_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_agent_pool(transfer.GetAgentPoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [transfer.ListAgentPoolsRequest, dict])
def test_list_agent_pools(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.return_value = transfer.ListAgentPoolsResponse(next_page_token='next_page_token_value')
        response = client.list_agent_pools(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ListAgentPoolsRequest()
    assert isinstance(response, pagers.ListAgentPoolsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_agent_pools_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        client.list_agent_pools()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ListAgentPoolsRequest()

@pytest.mark.asyncio
async def test_list_agent_pools_async(transport: str='grpc_asyncio', request_type=transfer.ListAgentPoolsRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.ListAgentPoolsResponse(next_page_token='next_page_token_value'))
        response = await client.list_agent_pools(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.ListAgentPoolsRequest()
    assert isinstance(response, pagers.ListAgentPoolsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_agent_pools_async_from_dict():
    await test_list_agent_pools_async(request_type=dict)

def test_list_agent_pools_field_headers():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.ListAgentPoolsRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.return_value = transfer.ListAgentPoolsResponse()
        client.list_agent_pools(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_agent_pools_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.ListAgentPoolsRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.ListAgentPoolsResponse())
        await client.list_agent_pools(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

def test_list_agent_pools_flattened():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.return_value = transfer.ListAgentPoolsResponse()
        client.list_agent_pools(project_id='project_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val

def test_list_agent_pools_flattened_error():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_agent_pools(transfer.ListAgentPoolsRequest(), project_id='project_id_value')

@pytest.mark.asyncio
async def test_list_agent_pools_flattened_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.return_value = transfer.ListAgentPoolsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.ListAgentPoolsResponse())
        response = await client.list_agent_pools(project_id='project_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_agent_pools_flattened_error_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_agent_pools(transfer.ListAgentPoolsRequest(), project_id='project_id_value')

def test_list_agent_pools_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.side_effect = (transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool(), transfer_types.AgentPool()], next_page_token='abc'), transfer.ListAgentPoolsResponse(agent_pools=[], next_page_token='def'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool()], next_page_token='ghi'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('project_id', ''),)),)
        pager = client.list_agent_pools(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer_types.AgentPool) for i in results))

def test_list_agent_pools_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__') as call:
        call.side_effect = (transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool(), transfer_types.AgentPool()], next_page_token='abc'), transfer.ListAgentPoolsResponse(agent_pools=[], next_page_token='def'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool()], next_page_token='ghi'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool()]), RuntimeError)
        pages = list(client.list_agent_pools(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_agent_pools_async_pager():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool(), transfer_types.AgentPool()], next_page_token='abc'), transfer.ListAgentPoolsResponse(agent_pools=[], next_page_token='def'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool()], next_page_token='ghi'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool()]), RuntimeError)
        async_pager = await client.list_agent_pools(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, transfer_types.AgentPool) for i in responses))

@pytest.mark.asyncio
async def test_list_agent_pools_async_pages():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_agent_pools), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool(), transfer_types.AgentPool()], next_page_token='abc'), transfer.ListAgentPoolsResponse(agent_pools=[], next_page_token='def'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool()], next_page_token='ghi'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_agent_pools(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [transfer.DeleteAgentPoolRequest, dict])
def test_delete_agent_pool(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        call.return_value = None
        response = client.delete_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.DeleteAgentPoolRequest()
    assert response is None

def test_delete_agent_pool_empty_call():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        client.delete_agent_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.DeleteAgentPoolRequest()

@pytest.mark.asyncio
async def test_delete_agent_pool_async(transport: str='grpc_asyncio', request_type=transfer.DeleteAgentPoolRequest):
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transfer.DeleteAgentPoolRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_agent_pool_async_from_dict():
    await test_delete_agent_pool_async(request_type=dict)

def test_delete_agent_pool_field_headers():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.DeleteAgentPoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        call.return_value = None
        client.delete_agent_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_agent_pool_field_headers_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transfer.DeleteAgentPoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_agent_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_agent_pool_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        call.return_value = None
        client.delete_agent_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_agent_pool_flattened_error():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_agent_pool(transfer.DeleteAgentPoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_agent_pool_flattened_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_agent_pool), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_agent_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_agent_pool_flattened_error_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_agent_pool(transfer.DeleteAgentPoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [transfer.GetGoogleServiceAccountRequest, dict])
def test_get_google_service_account_rest(request_type):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.GoogleServiceAccount(account_email='account_email_value', subject_id='subject_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.GoogleServiceAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_google_service_account(request)
    assert isinstance(response, transfer_types.GoogleServiceAccount)
    assert response.account_email == 'account_email_value'
    assert response.subject_id == 'subject_id_value'

def test_get_google_service_account_rest_required_fields(request_type=transfer.GetGoogleServiceAccountRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_google_service_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_google_service_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.GoogleServiceAccount()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.GoogleServiceAccount.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_google_service_account(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_google_service_account_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_google_service_account._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('projectId',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_google_service_account_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_get_google_service_account') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_get_google_service_account') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.GetGoogleServiceAccountRequest.pb(transfer.GetGoogleServiceAccountRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.GoogleServiceAccount.to_json(transfer_types.GoogleServiceAccount())
        request = transfer.GetGoogleServiceAccountRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.GoogleServiceAccount()
        client.get_google_service_account(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_google_service_account_rest_bad_request(transport: str='rest', request_type=transfer.GetGoogleServiceAccountRequest):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_google_service_account(request)

def test_get_google_service_account_rest_error():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.CreateTransferJobRequest, dict])
def test_create_transfer_job_rest(request_type):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request_init['transfer_job'] = {'name': 'name_value', 'description': 'description_value', 'project_id': 'project_id_value', 'transfer_spec': {'gcs_data_sink': {'bucket_name': 'bucket_name_value', 'path': 'path_value'}, 'posix_data_sink': {'root_directory': 'root_directory_value'}, 'gcs_data_source': {}, 'aws_s3_data_source': {'bucket_name': 'bucket_name_value', 'aws_access_key': {'access_key_id': 'access_key_id_value', 'secret_access_key': 'secret_access_key_value'}, 'path': 'path_value', 'role_arn': 'role_arn_value', 'credentials_secret': 'credentials_secret_value'}, 'http_data_source': {'list_url': 'list_url_value'}, 'posix_data_source': {}, 'azure_blob_storage_data_source': {'storage_account': 'storage_account_value', 'azure_credentials': {'sas_token': 'sas_token_value'}, 'container': 'container_value', 'path': 'path_value', 'credentials_secret': 'credentials_secret_value'}, 'aws_s3_compatible_data_source': {'bucket_name': 'bucket_name_value', 'path': 'path_value', 'endpoint': 'endpoint_value', 'region': 'region_value', 's3_metadata': {'auth_method': 1, 'request_model': 1, 'protocol': 1, 'list_api': 1}}, 'gcs_intermediate_data_location': {}, 'object_conditions': {'min_time_elapsed_since_last_modification': {'seconds': 751, 'nanos': 543}, 'max_time_elapsed_since_last_modification': {}, 'include_prefixes': ['include_prefixes_value1', 'include_prefixes_value2'], 'exclude_prefixes': ['exclude_prefixes_value1', 'exclude_prefixes_value2'], 'last_modified_since': {'seconds': 751, 'nanos': 543}, 'last_modified_before': {}}, 'transfer_options': {'overwrite_objects_already_existing_in_sink': True, 'delete_objects_unique_in_sink': True, 'delete_objects_from_source_after_transfer': True, 'overwrite_when': 1, 'metadata_options': {'symlink': 1, 'mode': 1, 'gid': 1, 'uid': 1, 'acl': 1, 'storage_class': 1, 'temporary_hold': 1, 'kms_key': 1, 'time_created': 1}}, 'transfer_manifest': {'location': 'location_value'}, 'source_agent_pool_name': 'source_agent_pool_name_value', 'sink_agent_pool_name': 'sink_agent_pool_name_value'}, 'notification_config': {'pubsub_topic': 'pubsub_topic_value', 'event_types': [1], 'payload_format': 1}, 'logging_config': {'log_actions': [1], 'log_action_states': [1], 'enable_onprem_gcs_transfer_logs': True}, 'schedule': {'schedule_start_date': {'year': 433, 'month': 550, 'day': 318}, 'schedule_end_date': {}, 'start_time_of_day': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'end_time_of_day': {}, 'repeat_interval': {}}, 'event_stream': {'name': 'name_value', 'event_stream_start_time': {}, 'event_stream_expiration_time': {}}, 'status': 1, 'creation_time': {}, 'last_modification_time': {}, 'deletion_time': {}, 'latest_operation_name': 'latest_operation_name_value'}
    test_field = transfer.CreateTransferJobRequest.meta.fields['transfer_job']

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
    for (field, value) in request_init['transfer_job'].items():
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
                for i in range(0, len(request_init['transfer_job'][field])):
                    del request_init['transfer_job'][field][i][subfield]
            else:
                del request_init['transfer_job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.TransferJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_transfer_job(request)
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

def test_create_transfer_job_rest_required_fields(request_type=transfer.CreateTransferJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.TransferJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.TransferJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_transfer_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_transfer_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_transfer_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('transferJob',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_transfer_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_create_transfer_job') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_create_transfer_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.CreateTransferJobRequest.pb(transfer.CreateTransferJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.TransferJob.to_json(transfer_types.TransferJob())
        request = transfer.CreateTransferJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.TransferJob()
        client.create_transfer_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_transfer_job_rest_bad_request(transport: str='rest', request_type=transfer.CreateTransferJobRequest):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_transfer_job(request)

def test_create_transfer_job_rest_error():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.UpdateTransferJobRequest, dict])
def test_update_transfer_job_rest(request_type):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.TransferJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_transfer_job(request)
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

def test_update_transfer_job_rest_required_fields(request_type=transfer.UpdateTransferJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['job_name'] = ''
    request_init['project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['jobName'] = 'job_name_value'
    jsonified_request['projectId'] = 'project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'jobName' in jsonified_request
    assert jsonified_request['jobName'] == 'job_name_value'
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.TransferJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.TransferJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_transfer_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_transfer_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_transfer_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('jobName', 'projectId', 'transferJob'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_transfer_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_update_transfer_job') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_update_transfer_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.UpdateTransferJobRequest.pb(transfer.UpdateTransferJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.TransferJob.to_json(transfer_types.TransferJob())
        request = transfer.UpdateTransferJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.TransferJob()
        client.update_transfer_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_transfer_job_rest_bad_request(transport: str='rest', request_type=transfer.UpdateTransferJobRequest):
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_transfer_job(request)

def test_update_transfer_job_rest_error():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.GetTransferJobRequest, dict])
def test_get_transfer_job_rest(request_type):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.TransferJob(name='name_value', description='description_value', project_id='project_id_value', status=transfer_types.TransferJob.Status.ENABLED, latest_operation_name='latest_operation_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.TransferJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_transfer_job(request)
    assert isinstance(response, transfer_types.TransferJob)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.project_id == 'project_id_value'
    assert response.status == transfer_types.TransferJob.Status.ENABLED
    assert response.latest_operation_name == 'latest_operation_name_value'

def test_get_transfer_job_rest_required_fields(request_type=transfer.GetTransferJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['job_name'] = ''
    request_init['project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'projectId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == request_init['project_id']
    jsonified_request['jobName'] = 'job_name_value'
    jsonified_request['projectId'] = 'project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transfer_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('project_id',))
    jsonified_request.update(unset_fields)
    assert 'jobName' in jsonified_request
    assert jsonified_request['jobName'] == 'job_name_value'
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.TransferJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.TransferJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_transfer_job(request)
            expected_params = [('projectId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_transfer_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_transfer_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('projectId',)) & set(('jobName', 'projectId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_transfer_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_get_transfer_job') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_get_transfer_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.GetTransferJobRequest.pb(transfer.GetTransferJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.TransferJob.to_json(transfer_types.TransferJob())
        request = transfer.GetTransferJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.TransferJob()
        client.get_transfer_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_transfer_job_rest_bad_request(transport: str='rest', request_type=transfer.GetTransferJobRequest):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_transfer_job(request)

def test_get_transfer_job_rest_error():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.ListTransferJobsRequest, dict])
def test_list_transfer_jobs_rest(request_type):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.ListTransferJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.ListTransferJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_transfer_jobs(request)
    assert isinstance(response, pagers.ListTransferJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_jobs_rest_required_fields(request_type=transfer.ListTransferJobsRequest):
    if False:
        return 10
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['filter'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'filter' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == request_init['filter']
    jsonified_request['filter'] = 'filter_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == 'filter_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer.ListTransferJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer.ListTransferJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_transfer_jobs(request)
            expected_params = [('filter', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_transfer_jobs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_transfer_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('filter',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_transfer_jobs_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_list_transfer_jobs') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_list_transfer_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.ListTransferJobsRequest.pb(transfer.ListTransferJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer.ListTransferJobsResponse.to_json(transfer.ListTransferJobsResponse())
        request = transfer.ListTransferJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer.ListTransferJobsResponse()
        client.list_transfer_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_transfer_jobs_rest_bad_request(transport: str='rest', request_type=transfer.ListTransferJobsRequest):
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_transfer_jobs(request)

def test_list_transfer_jobs_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob(), transfer_types.TransferJob()], next_page_token='abc'), transfer.ListTransferJobsResponse(transfer_jobs=[], next_page_token='def'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob()], next_page_token='ghi'), transfer.ListTransferJobsResponse(transfer_jobs=[transfer_types.TransferJob(), transfer_types.TransferJob()]))
        response = response + response
        response = tuple((transfer.ListTransferJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_transfer_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer_types.TransferJob) for i in results))
        pages = list(client.list_transfer_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [transfer.PauseTransferOperationRequest, dict])
def test_pause_transfer_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'transferOperations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.pause_transfer_operation(request)
    assert response is None

def test_pause_transfer_operation_rest_required_fields(request_type=transfer.PauseTransferOperationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_transfer_operation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_transfer_operation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.pause_transfer_operation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_pause_transfer_operation_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.pause_transfer_operation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_pause_transfer_operation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_pause_transfer_operation') as pre:
        pre.assert_not_called()
        pb_message = transfer.PauseTransferOperationRequest.pb(transfer.PauseTransferOperationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = transfer.PauseTransferOperationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.pause_transfer_operation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_pause_transfer_operation_rest_bad_request(transport: str='rest', request_type=transfer.PauseTransferOperationRequest):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'transferOperations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.pause_transfer_operation(request)

def test_pause_transfer_operation_rest_error():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.ResumeTransferOperationRequest, dict])
def test_resume_transfer_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'transferOperations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_transfer_operation(request)
    assert response is None

def test_resume_transfer_operation_rest_required_fields(request_type=transfer.ResumeTransferOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_transfer_operation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_transfer_operation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.resume_transfer_operation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_transfer_operation_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_transfer_operation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_transfer_operation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_resume_transfer_operation') as pre:
        pre.assert_not_called()
        pb_message = transfer.ResumeTransferOperationRequest.pb(transfer.ResumeTransferOperationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = transfer.ResumeTransferOperationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.resume_transfer_operation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_resume_transfer_operation_rest_bad_request(transport: str='rest', request_type=transfer.ResumeTransferOperationRequest):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'transferOperations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_transfer_operation(request)

def test_resume_transfer_operation_rest_error():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.RunTransferJobRequest, dict])
def test_run_transfer_job_rest(request_type):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.run_transfer_job(request)
    assert response.operation.name == 'operations/spam'

def test_run_transfer_job_rest_required_fields(request_type=transfer.RunTransferJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['job_name'] = ''
    request_init['project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['jobName'] = 'job_name_value'
    jsonified_request['projectId'] = 'project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'jobName' in jsonified_request
    assert jsonified_request['jobName'] == 'job_name_value'
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.run_transfer_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_run_transfer_job_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.run_transfer_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('jobName', 'projectId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_run_transfer_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_run_transfer_job') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_run_transfer_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.RunTransferJobRequest.pb(transfer.RunTransferJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = transfer.RunTransferJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.run_transfer_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_run_transfer_job_rest_bad_request(transport: str='rest', request_type=transfer.RunTransferJobRequest):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.run_transfer_job(request)

def test_run_transfer_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.DeleteTransferJobRequest, dict])
def test_delete_transfer_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_transfer_job(request)
    assert response is None

def test_delete_transfer_job_rest_required_fields(request_type=transfer.DeleteTransferJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['job_name'] = ''
    request_init['project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'projectId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transfer_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == request_init['project_id']
    jsonified_request['jobName'] = 'job_name_value'
    jsonified_request['projectId'] = 'project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transfer_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('project_id',))
    jsonified_request.update(unset_fields)
    assert 'jobName' in jsonified_request
    assert jsonified_request['jobName'] == 'job_name_value'
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_transfer_job(request)
            expected_params = [('projectId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_transfer_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_transfer_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('projectId',)) & set(('jobName', 'projectId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_transfer_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_delete_transfer_job') as pre:
        pre.assert_not_called()
        pb_message = transfer.DeleteTransferJobRequest.pb(transfer.DeleteTransferJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = transfer.DeleteTransferJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_transfer_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_transfer_job_rest_bad_request(transport: str='rest', request_type=transfer.DeleteTransferJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'job_name': 'transferJobs/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_transfer_job(request)

def test_delete_transfer_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.CreateAgentPoolRequest, dict])
def test_create_agent_pool_rest(request_type):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1'}
    request_init['agent_pool'] = {'name': 'name_value', 'display_name': 'display_name_value', 'state': 1, 'bandwidth_limit': {'limit_mbps': 1072}}
    test_field = transfer.CreateAgentPoolRequest.meta.fields['agent_pool']

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
    for (field, value) in request_init['agent_pool'].items():
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
                for i in range(0, len(request_init['agent_pool'][field])):
                    del request_init['agent_pool'][field][i][subfield]
            else:
                del request_init['agent_pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.AgentPool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_agent_pool(request)
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

def test_create_agent_pool_rest_required_fields(request_type=transfer.CreateAgentPoolRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['agent_pool_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'agentPoolId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_agent_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'agentPoolId' in jsonified_request
    assert jsonified_request['agentPoolId'] == request_init['agent_pool_id']
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['agentPoolId'] = 'agent_pool_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_agent_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('agent_pool_id',))
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'agentPoolId' in jsonified_request
    assert jsonified_request['agentPoolId'] == 'agent_pool_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.AgentPool()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.AgentPool.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_agent_pool(request)
            expected_params = [('agentPoolId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_agent_pool_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_agent_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('agentPoolId',)) & set(('projectId', 'agentPool', 'agentPoolId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_agent_pool_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_create_agent_pool') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_create_agent_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.CreateAgentPoolRequest.pb(transfer.CreateAgentPoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.AgentPool.to_json(transfer_types.AgentPool())
        request = transfer.CreateAgentPoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.AgentPool()
        client.create_agent_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_agent_pool_rest_bad_request(transport: str='rest', request_type=transfer.CreateAgentPoolRequest):
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_agent_pool(request)

def test_create_agent_pool_rest_flattened():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.AgentPool()
        sample_request = {'project_id': 'sample1'}
        mock_args = dict(project_id='project_id_value', agent_pool=transfer_types.AgentPool(name='name_value'), agent_pool_id='agent_pool_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.AgentPool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_agent_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id=*}/agentPools' % client.transport._host, args[1])

def test_create_agent_pool_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_agent_pool(transfer.CreateAgentPoolRequest(), project_id='project_id_value', agent_pool=transfer_types.AgentPool(name='name_value'), agent_pool_id='agent_pool_id_value')

def test_create_agent_pool_rest_error():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.UpdateAgentPoolRequest, dict])
def test_update_agent_pool_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'agent_pool': {'name': 'projects/sample1/agentPools/sample2'}}
    request_init['agent_pool'] = {'name': 'projects/sample1/agentPools/sample2', 'display_name': 'display_name_value', 'state': 1, 'bandwidth_limit': {'limit_mbps': 1072}}
    test_field = transfer.UpdateAgentPoolRequest.meta.fields['agent_pool']

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
    for (field, value) in request_init['agent_pool'].items():
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
                for i in range(0, len(request_init['agent_pool'][field])):
                    del request_init['agent_pool'][field][i][subfield]
            else:
                del request_init['agent_pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.AgentPool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_agent_pool(request)
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

def test_update_agent_pool_rest_required_fields(request_type=transfer.UpdateAgentPoolRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_agent_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_agent_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.AgentPool()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.AgentPool.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_agent_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_agent_pool_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_agent_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('agentPool',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_agent_pool_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_update_agent_pool') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_update_agent_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.UpdateAgentPoolRequest.pb(transfer.UpdateAgentPoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.AgentPool.to_json(transfer_types.AgentPool())
        request = transfer.UpdateAgentPoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.AgentPool()
        client.update_agent_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_agent_pool_rest_bad_request(transport: str='rest', request_type=transfer.UpdateAgentPoolRequest):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'agent_pool': {'name': 'projects/sample1/agentPools/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_agent_pool(request)

def test_update_agent_pool_rest_flattened():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.AgentPool()
        sample_request = {'agent_pool': {'name': 'projects/sample1/agentPools/sample2'}}
        mock_args = dict(agent_pool=transfer_types.AgentPool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.AgentPool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_agent_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{agent_pool.name=projects/*/agentPools/*}' % client.transport._host, args[1])

def test_update_agent_pool_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_agent_pool(transfer.UpdateAgentPoolRequest(), agent_pool=transfer_types.AgentPool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_agent_pool_rest_error():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.GetAgentPoolRequest, dict])
def test_get_agent_pool_rest(request_type):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/agentPools/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.AgentPool(name='name_value', display_name='display_name_value', state=transfer_types.AgentPool.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.AgentPool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_agent_pool(request)
    assert isinstance(response, transfer_types.AgentPool)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == transfer_types.AgentPool.State.CREATING

def test_get_agent_pool_rest_required_fields(request_type=transfer.GetAgentPoolRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_agent_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_agent_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer_types.AgentPool()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer_types.AgentPool.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_agent_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_agent_pool_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_agent_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_agent_pool_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_get_agent_pool') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_get_agent_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.GetAgentPoolRequest.pb(transfer.GetAgentPoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer_types.AgentPool.to_json(transfer_types.AgentPool())
        request = transfer.GetAgentPoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer_types.AgentPool()
        client.get_agent_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_agent_pool_rest_bad_request(transport: str='rest', request_type=transfer.GetAgentPoolRequest):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/agentPools/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_agent_pool(request)

def test_get_agent_pool_rest_flattened():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer_types.AgentPool()
        sample_request = {'name': 'projects/sample1/agentPools/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer_types.AgentPool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_agent_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/agentPools/*}' % client.transport._host, args[1])

def test_get_agent_pool_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_agent_pool(transfer.GetAgentPoolRequest(), name='name_value')

def test_get_agent_pool_rest_error():
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transfer.ListAgentPoolsRequest, dict])
def test_list_agent_pools_rest(request_type):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.ListAgentPoolsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.ListAgentPoolsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_agent_pools(request)
    assert isinstance(response, pagers.ListAgentPoolsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_agent_pools_rest_required_fields(request_type=transfer.ListAgentPoolsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_agent_pools._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_agent_pools._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer.ListAgentPoolsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer.ListAgentPoolsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_agent_pools(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_agent_pools_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_agent_pools._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('projectId',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_agent_pools_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'post_list_agent_pools') as post, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_list_agent_pools') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transfer.ListAgentPoolsRequest.pb(transfer.ListAgentPoolsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer.ListAgentPoolsResponse.to_json(transfer.ListAgentPoolsResponse())
        request = transfer.ListAgentPoolsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer.ListAgentPoolsResponse()
        client.list_agent_pools(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_agent_pools_rest_bad_request(transport: str='rest', request_type=transfer.ListAgentPoolsRequest):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_agent_pools(request)

def test_list_agent_pools_rest_flattened():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.ListAgentPoolsResponse()
        sample_request = {'project_id': 'sample1'}
        mock_args = dict(project_id='project_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.ListAgentPoolsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_agent_pools(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id=*}/agentPools' % client.transport._host, args[1])

def test_list_agent_pools_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_agent_pools(transfer.ListAgentPoolsRequest(), project_id='project_id_value')

def test_list_agent_pools_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool(), transfer_types.AgentPool()], next_page_token='abc'), transfer.ListAgentPoolsResponse(agent_pools=[], next_page_token='def'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool()], next_page_token='ghi'), transfer.ListAgentPoolsResponse(agent_pools=[transfer_types.AgentPool(), transfer_types.AgentPool()]))
        response = response + response
        response = tuple((transfer.ListAgentPoolsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project_id': 'sample1'}
        pager = client.list_agent_pools(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer_types.AgentPool) for i in results))
        pages = list(client.list_agent_pools(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [transfer.DeleteAgentPoolRequest, dict])
def test_delete_agent_pool_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/agentPools/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_agent_pool(request)
    assert response is None

def test_delete_agent_pool_rest_required_fields(request_type=transfer.DeleteAgentPoolRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.StorageTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_agent_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_agent_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_agent_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_agent_pool_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_agent_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_agent_pool_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.StorageTransferServiceRestInterceptor())
    client = StorageTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.StorageTransferServiceRestInterceptor, 'pre_delete_agent_pool') as pre:
        pre.assert_not_called()
        pb_message = transfer.DeleteAgentPoolRequest.pb(transfer.DeleteAgentPoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = transfer.DeleteAgentPoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_agent_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_agent_pool_rest_bad_request(transport: str='rest', request_type=transfer.DeleteAgentPoolRequest):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/agentPools/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_agent_pool(request)

def test_delete_agent_pool_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/agentPools/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_agent_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/agentPools/*}' % client.transport._host, args[1])

def test_delete_agent_pool_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_agent_pool(transfer.DeleteAgentPoolRequest(), name='name_value')

def test_delete_agent_pool_rest_error():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.StorageTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.StorageTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = StorageTransferServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.StorageTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = StorageTransferServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = StorageTransferServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.StorageTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = StorageTransferServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.StorageTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = StorageTransferServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.StorageTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.StorageTransferServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceGrpcAsyncIOTransport, transports.StorageTransferServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = StorageTransferServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.StorageTransferServiceGrpcTransport)

def test_storage_transfer_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.StorageTransferServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_storage_transfer_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.storage_transfer_v1.services.storage_transfer_service.transports.StorageTransferServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.StorageTransferServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_google_service_account', 'create_transfer_job', 'update_transfer_job', 'get_transfer_job', 'list_transfer_jobs', 'pause_transfer_operation', 'resume_transfer_operation', 'run_transfer_job', 'delete_transfer_job', 'create_agent_pool', 'update_agent_pool', 'get_agent_pool', 'list_agent_pools', 'delete_agent_pool', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_storage_transfer_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.storage_transfer_v1.services.storage_transfer_service.transports.StorageTransferServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.StorageTransferServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_storage_transfer_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.storage_transfer_v1.services.storage_transfer_service.transports.StorageTransferServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.StorageTransferServiceTransport()
        adc.assert_called_once()

def test_storage_transfer_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        StorageTransferServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceGrpcAsyncIOTransport])
def test_storage_transfer_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceGrpcAsyncIOTransport, transports.StorageTransferServiceRestTransport])
def test_storage_transfer_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.StorageTransferServiceGrpcTransport, grpc_helpers), (transports.StorageTransferServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_storage_transfer_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('storagetransfer.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='storagetransfer.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceGrpcAsyncIOTransport])
def test_storage_transfer_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_storage_transfer_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.StorageTransferServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_storage_transfer_service_rest_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_storage_transfer_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='storagetransfer.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('storagetransfer.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://storagetransfer.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_storage_transfer_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='storagetransfer.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('storagetransfer.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://storagetransfer.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_storage_transfer_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = StorageTransferServiceClient(credentials=creds1, transport=transport_name)
    client2 = StorageTransferServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_google_service_account._session
    session2 = client2.transport.get_google_service_account._session
    assert session1 != session2
    session1 = client1.transport.create_transfer_job._session
    session2 = client2.transport.create_transfer_job._session
    assert session1 != session2
    session1 = client1.transport.update_transfer_job._session
    session2 = client2.transport.update_transfer_job._session
    assert session1 != session2
    session1 = client1.transport.get_transfer_job._session
    session2 = client2.transport.get_transfer_job._session
    assert session1 != session2
    session1 = client1.transport.list_transfer_jobs._session
    session2 = client2.transport.list_transfer_jobs._session
    assert session1 != session2
    session1 = client1.transport.pause_transfer_operation._session
    session2 = client2.transport.pause_transfer_operation._session
    assert session1 != session2
    session1 = client1.transport.resume_transfer_operation._session
    session2 = client2.transport.resume_transfer_operation._session
    assert session1 != session2
    session1 = client1.transport.run_transfer_job._session
    session2 = client2.transport.run_transfer_job._session
    assert session1 != session2
    session1 = client1.transport.delete_transfer_job._session
    session2 = client2.transport.delete_transfer_job._session
    assert session1 != session2
    session1 = client1.transport.create_agent_pool._session
    session2 = client2.transport.create_agent_pool._session
    assert session1 != session2
    session1 = client1.transport.update_agent_pool._session
    session2 = client2.transport.update_agent_pool._session
    assert session1 != session2
    session1 = client1.transport.get_agent_pool._session
    session2 = client2.transport.get_agent_pool._session
    assert session1 != session2
    session1 = client1.transport.list_agent_pools._session
    session2 = client2.transport.list_agent_pools._session
    assert session1 != session2
    session1 = client1.transport.delete_agent_pool._session
    session2 = client2.transport.delete_agent_pool._session
    assert session1 != session2

def test_storage_transfer_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.StorageTransferServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_storage_transfer_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.StorageTransferServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceGrpcAsyncIOTransport])
def test_storage_transfer_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.StorageTransferServiceGrpcTransport, transports.StorageTransferServiceGrpcAsyncIOTransport])
def test_storage_transfer_service_transport_channel_mtls_with_adc(transport_class):
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

def test_storage_transfer_service_grpc_lro_client():
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_storage_transfer_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_agent_pools_path():
    if False:
        print('Hello World!')
    project_id = 'squid'
    agent_pool_id = 'clam'
    expected = 'projects/{project_id}/agentPools/{agent_pool_id}'.format(project_id=project_id, agent_pool_id=agent_pool_id)
    actual = StorageTransferServiceClient.agent_pools_path(project_id, agent_pool_id)
    assert expected == actual

def test_parse_agent_pools_path():
    if False:
        return 10
    expected = {'project_id': 'whelk', 'agent_pool_id': 'octopus'}
    path = StorageTransferServiceClient.agent_pools_path(**expected)
    actual = StorageTransferServiceClient.parse_agent_pools_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = StorageTransferServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'nudibranch'}
    path = StorageTransferServiceClient.common_billing_account_path(**expected)
    actual = StorageTransferServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = StorageTransferServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'mussel'}
    path = StorageTransferServiceClient.common_folder_path(**expected)
    actual = StorageTransferServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = StorageTransferServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nautilus'}
    path = StorageTransferServiceClient.common_organization_path(**expected)
    actual = StorageTransferServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = StorageTransferServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone'}
    path = StorageTransferServiceClient.common_project_path(**expected)
    actual = StorageTransferServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = StorageTransferServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = StorageTransferServiceClient.common_location_path(**expected)
    actual = StorageTransferServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.StorageTransferServiceTransport, '_prep_wrapped_messages') as prep:
        client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.StorageTransferServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = StorageTransferServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'transferOperations/sample1'}, request)
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'transferOperations/sample1'}
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
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'transferOperations/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        print('Hello World!')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'transferOperations/sample1'}
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
        for i in range(10):
            print('nop')
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'transferOperations'}, request)
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'transferOperations'}
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
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = StorageTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = StorageTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(StorageTransferServiceClient, transports.StorageTransferServiceGrpcTransport), (StorageTransferServiceAsyncClient, transports.StorageTransferServiceGrpcAsyncIOTransport)])
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