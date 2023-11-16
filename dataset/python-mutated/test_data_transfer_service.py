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
from google.cloud.location import locations_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.bigquery_datatransfer_v1.services.data_transfer_service import DataTransferServiceAsyncClient, DataTransferServiceClient, pagers, transports
from google.cloud.bigquery_datatransfer_v1.types import datatransfer, transfer

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert DataTransferServiceClient._get_default_mtls_endpoint(None) is None
    assert DataTransferServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DataTransferServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DataTransferServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DataTransferServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DataTransferServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DataTransferServiceClient, 'grpc'), (DataTransferServiceAsyncClient, 'grpc_asyncio'), (DataTransferServiceClient, 'rest')])
def test_data_transfer_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('bigquerydatatransfer.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigquerydatatransfer.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DataTransferServiceGrpcTransport, 'grpc'), (transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.DataTransferServiceRestTransport, 'rest')])
def test_data_transfer_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DataTransferServiceClient, 'grpc'), (DataTransferServiceAsyncClient, 'grpc_asyncio'), (DataTransferServiceClient, 'rest')])
def test_data_transfer_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('bigquerydatatransfer.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigquerydatatransfer.googleapis.com')

def test_data_transfer_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = DataTransferServiceClient.get_transport_class()
    available_transports = [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceRestTransport]
    assert transport in available_transports
    transport = DataTransferServiceClient.get_transport_class('grpc')
    assert transport == transports.DataTransferServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DataTransferServiceClient, transports.DataTransferServiceGrpcTransport, 'grpc'), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DataTransferServiceClient, transports.DataTransferServiceRestTransport, 'rest')])
@mock.patch.object(DataTransferServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTransferServiceClient))
@mock.patch.object(DataTransferServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTransferServiceAsyncClient))
def test_data_transfer_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(DataTransferServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DataTransferServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DataTransferServiceClient, transports.DataTransferServiceGrpcTransport, 'grpc', 'true'), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DataTransferServiceClient, transports.DataTransferServiceGrpcTransport, 'grpc', 'false'), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (DataTransferServiceClient, transports.DataTransferServiceRestTransport, 'rest', 'true'), (DataTransferServiceClient, transports.DataTransferServiceRestTransport, 'rest', 'false')])
@mock.patch.object(DataTransferServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTransferServiceClient))
@mock.patch.object(DataTransferServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTransferServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_data_transfer_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [DataTransferServiceClient, DataTransferServiceAsyncClient])
@mock.patch.object(DataTransferServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTransferServiceClient))
@mock.patch.object(DataTransferServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DataTransferServiceAsyncClient))
def test_data_transfer_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DataTransferServiceClient, transports.DataTransferServiceGrpcTransport, 'grpc'), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DataTransferServiceClient, transports.DataTransferServiceRestTransport, 'rest')])
def test_data_transfer_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DataTransferServiceClient, transports.DataTransferServiceGrpcTransport, 'grpc', grpc_helpers), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (DataTransferServiceClient, transports.DataTransferServiceRestTransport, 'rest', None)])
def test_data_transfer_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_data_transfer_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.bigquery_datatransfer_v1.services.data_transfer_service.transports.DataTransferServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DataTransferServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DataTransferServiceClient, transports.DataTransferServiceGrpcTransport, 'grpc', grpc_helpers), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_data_transfer_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('bigquerydatatransfer.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='bigquerydatatransfer.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [datatransfer.GetDataSourceRequest, dict])
def test_get_data_source(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        call.return_value = datatransfer.DataSource(name='name_value', data_source_id='data_source_id_value', display_name='display_name_value', description='description_value', client_id='client_id_value', scopes=['scopes_value'], transfer_type=transfer.TransferType.BATCH, supports_multiple_transfers=True, update_deadline_seconds=2406, default_schedule='default_schedule_value', supports_custom_schedule=True, help_url='help_url_value', authorization_type=datatransfer.DataSource.AuthorizationType.AUTHORIZATION_CODE, data_refresh_type=datatransfer.DataSource.DataRefreshType.SLIDING_WINDOW, default_data_refresh_window_days=3379, manual_runs_disabled=True)
        response = client.get_data_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetDataSourceRequest()
    assert isinstance(response, datatransfer.DataSource)
    assert response.name == 'name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.client_id == 'client_id_value'
    assert response.scopes == ['scopes_value']
    assert response.transfer_type == transfer.TransferType.BATCH
    assert response.supports_multiple_transfers is True
    assert response.update_deadline_seconds == 2406
    assert response.default_schedule == 'default_schedule_value'
    assert response.supports_custom_schedule is True
    assert response.help_url == 'help_url_value'
    assert response.authorization_type == datatransfer.DataSource.AuthorizationType.AUTHORIZATION_CODE
    assert response.data_refresh_type == datatransfer.DataSource.DataRefreshType.SLIDING_WINDOW
    assert response.default_data_refresh_window_days == 3379
    assert response.manual_runs_disabled is True

def test_get_data_source_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        client.get_data_source()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetDataSourceRequest()

@pytest.mark.asyncio
async def test_get_data_source_async(transport: str='grpc_asyncio', request_type=datatransfer.GetDataSourceRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.DataSource(name='name_value', data_source_id='data_source_id_value', display_name='display_name_value', description='description_value', client_id='client_id_value', scopes=['scopes_value'], transfer_type=transfer.TransferType.BATCH, supports_multiple_transfers=True, update_deadline_seconds=2406, default_schedule='default_schedule_value', supports_custom_schedule=True, help_url='help_url_value', authorization_type=datatransfer.DataSource.AuthorizationType.AUTHORIZATION_CODE, data_refresh_type=datatransfer.DataSource.DataRefreshType.SLIDING_WINDOW, default_data_refresh_window_days=3379, manual_runs_disabled=True))
        response = await client.get_data_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetDataSourceRequest()
    assert isinstance(response, datatransfer.DataSource)
    assert response.name == 'name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.client_id == 'client_id_value'
    assert response.scopes == ['scopes_value']
    assert response.transfer_type == transfer.TransferType.BATCH
    assert response.supports_multiple_transfers is True
    assert response.update_deadline_seconds == 2406
    assert response.default_schedule == 'default_schedule_value'
    assert response.supports_custom_schedule is True
    assert response.help_url == 'help_url_value'
    assert response.authorization_type == datatransfer.DataSource.AuthorizationType.AUTHORIZATION_CODE
    assert response.data_refresh_type == datatransfer.DataSource.DataRefreshType.SLIDING_WINDOW
    assert response.default_data_refresh_window_days == 3379
    assert response.manual_runs_disabled is True

@pytest.mark.asyncio
async def test_get_data_source_async_from_dict():
    await test_get_data_source_async(request_type=dict)

def test_get_data_source_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.GetDataSourceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        call.return_value = datatransfer.DataSource()
        client.get_data_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_data_source_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.GetDataSourceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.DataSource())
        await client.get_data_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_data_source_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        call.return_value = datatransfer.DataSource()
        client.get_data_source(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_data_source_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_data_source(datatransfer.GetDataSourceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_data_source_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_data_source), '__call__') as call:
        call.return_value = datatransfer.DataSource()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.DataSource())
        response = await client.get_data_source(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_data_source_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_data_source(datatransfer.GetDataSourceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [datatransfer.ListDataSourcesRequest, dict])
def test_list_data_sources(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.return_value = datatransfer.ListDataSourcesResponse(next_page_token='next_page_token_value')
        response = client.list_data_sources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListDataSourcesRequest()
    assert isinstance(response, pagers.ListDataSourcesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_data_sources_empty_call():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        client.list_data_sources()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListDataSourcesRequest()

@pytest.mark.asyncio
async def test_list_data_sources_async(transport: str='grpc_asyncio', request_type=datatransfer.ListDataSourcesRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListDataSourcesResponse(next_page_token='next_page_token_value'))
        response = await client.list_data_sources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListDataSourcesRequest()
    assert isinstance(response, pagers.ListDataSourcesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_data_sources_async_from_dict():
    await test_list_data_sources_async(request_type=dict)

def test_list_data_sources_field_headers():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListDataSourcesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.return_value = datatransfer.ListDataSourcesResponse()
        client.list_data_sources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_data_sources_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListDataSourcesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListDataSourcesResponse())
        await client.list_data_sources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_data_sources_flattened():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.return_value = datatransfer.ListDataSourcesResponse()
        client.list_data_sources(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_data_sources_flattened_error():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_data_sources(datatransfer.ListDataSourcesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_data_sources_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.return_value = datatransfer.ListDataSourcesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListDataSourcesResponse())
        response = await client.list_data_sources(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_data_sources_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_data_sources(datatransfer.ListDataSourcesRequest(), parent='parent_value')

def test_list_data_sources_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.side_effect = (datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource(), datatransfer.DataSource()], next_page_token='abc'), datatransfer.ListDataSourcesResponse(data_sources=[], next_page_token='def'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource()], next_page_token='ghi'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_data_sources(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, datatransfer.DataSource) for i in results))

def test_list_data_sources_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_data_sources), '__call__') as call:
        call.side_effect = (datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource(), datatransfer.DataSource()], next_page_token='abc'), datatransfer.ListDataSourcesResponse(data_sources=[], next_page_token='def'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource()], next_page_token='ghi'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource()]), RuntimeError)
        pages = list(client.list_data_sources(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_data_sources_async_pager():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_sources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource(), datatransfer.DataSource()], next_page_token='abc'), datatransfer.ListDataSourcesResponse(data_sources=[], next_page_token='def'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource()], next_page_token='ghi'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource()]), RuntimeError)
        async_pager = await client.list_data_sources(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, datatransfer.DataSource) for i in responses))

@pytest.mark.asyncio
async def test_list_data_sources_async_pages():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_data_sources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource(), datatransfer.DataSource()], next_page_token='abc'), datatransfer.ListDataSourcesResponse(data_sources=[], next_page_token='def'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource()], next_page_token='ghi'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_data_sources(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.CreateTransferConfigRequest, dict])
def test_create_transfer_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response = client.create_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.CreateTransferConfigRequest()
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_create_transfer_config_empty_call():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        client.create_transfer_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.CreateTransferConfigRequest()

@pytest.mark.asyncio
async def test_create_transfer_config_async(transport: str='grpc_asyncio', request_type=datatransfer.CreateTransferConfigRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value'))
        response = await client.create_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.CreateTransferConfigRequest()
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

@pytest.mark.asyncio
async def test_create_transfer_config_async_from_dict():
    await test_create_transfer_config_async(request_type=dict)

def test_create_transfer_config_field_headers():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.CreateTransferConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        client.create_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_transfer_config_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.CreateTransferConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig())
        await client.create_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_transfer_config_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        client.create_transfer_config(parent='parent_value', transfer_config=transfer.TransferConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].transfer_config
        mock_val = transfer.TransferConfig(name='name_value')
        assert arg == mock_val

def test_create_transfer_config_flattened_error():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_transfer_config(datatransfer.CreateTransferConfigRequest(), parent='parent_value', transfer_config=transfer.TransferConfig(name='name_value'))

@pytest.mark.asyncio
async def test_create_transfer_config_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig())
        response = await client.create_transfer_config(parent='parent_value', transfer_config=transfer.TransferConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].transfer_config
        mock_val = transfer.TransferConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_transfer_config_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_transfer_config(datatransfer.CreateTransferConfigRequest(), parent='parent_value', transfer_config=transfer.TransferConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [datatransfer.UpdateTransferConfigRequest, dict])
def test_update_transfer_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response = client.update_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.UpdateTransferConfigRequest()
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_update_transfer_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        client.update_transfer_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.UpdateTransferConfigRequest()

@pytest.mark.asyncio
async def test_update_transfer_config_async(transport: str='grpc_asyncio', request_type=datatransfer.UpdateTransferConfigRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value'))
        response = await client.update_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.UpdateTransferConfigRequest()
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

@pytest.mark.asyncio
async def test_update_transfer_config_async_from_dict():
    await test_update_transfer_config_async(request_type=dict)

def test_update_transfer_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.UpdateTransferConfigRequest()
    request.transfer_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        client.update_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'transfer_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_transfer_config_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.UpdateTransferConfigRequest()
    request.transfer_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig())
        await client.update_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'transfer_config.name=name_value') in kw['metadata']

def test_update_transfer_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        client.update_transfer_config(transfer_config=transfer.TransferConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].transfer_config
        mock_val = transfer.TransferConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_transfer_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_transfer_config(datatransfer.UpdateTransferConfigRequest(), transfer_config=transfer.TransferConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_transfer_config_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig())
        response = await client.update_transfer_config(transfer_config=transfer.TransferConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].transfer_config
        mock_val = transfer.TransferConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_transfer_config_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_transfer_config(datatransfer.UpdateTransferConfigRequest(), transfer_config=transfer.TransferConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [datatransfer.DeleteTransferConfigRequest, dict])
def test_delete_transfer_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        call.return_value = None
        response = client.delete_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.DeleteTransferConfigRequest()
    assert response is None

def test_delete_transfer_config_empty_call():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        client.delete_transfer_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.DeleteTransferConfigRequest()

@pytest.mark.asyncio
async def test_delete_transfer_config_async(transport: str='grpc_asyncio', request_type=datatransfer.DeleteTransferConfigRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.DeleteTransferConfigRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_transfer_config_async_from_dict():
    await test_delete_transfer_config_async(request_type=dict)

def test_delete_transfer_config_field_headers():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.DeleteTransferConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        call.return_value = None
        client.delete_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_transfer_config_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.DeleteTransferConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_transfer_config_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        call.return_value = None
        client.delete_transfer_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_transfer_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_transfer_config(datatransfer.DeleteTransferConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_transfer_config_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_transfer_config), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transfer_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_transfer_config_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_transfer_config(datatransfer.DeleteTransferConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [datatransfer.GetTransferConfigRequest, dict])
def test_get_transfer_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response = client.get_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetTransferConfigRequest()
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_get_transfer_config_empty_call():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        client.get_transfer_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetTransferConfigRequest()

@pytest.mark.asyncio
async def test_get_transfer_config_async(transport: str='grpc_asyncio', request_type=datatransfer.GetTransferConfigRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value'))
        response = await client.get_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetTransferConfigRequest()
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

@pytest.mark.asyncio
async def test_get_transfer_config_async_from_dict():
    await test_get_transfer_config_async(request_type=dict)

def test_get_transfer_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.GetTransferConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        client.get_transfer_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_transfer_config_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.GetTransferConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig())
        await client.get_transfer_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_transfer_config_flattened():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        client.get_transfer_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_transfer_config_flattened_error():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_transfer_config(datatransfer.GetTransferConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_transfer_config_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_transfer_config), '__call__') as call:
        call.return_value = transfer.TransferConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferConfig())
        response = await client.get_transfer_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_transfer_config_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_transfer_config(datatransfer.GetTransferConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [datatransfer.ListTransferConfigsRequest, dict])
def test_list_transfer_configs(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.return_value = datatransfer.ListTransferConfigsResponse(next_page_token='next_page_token_value')
        response = client.list_transfer_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferConfigsRequest()
    assert isinstance(response, pagers.ListTransferConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_configs_empty_call():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        client.list_transfer_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferConfigsRequest()

@pytest.mark.asyncio
async def test_list_transfer_configs_async(transport: str='grpc_asyncio', request_type=datatransfer.ListTransferConfigsRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferConfigsResponse(next_page_token='next_page_token_value'))
        response = await client.list_transfer_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferConfigsRequest()
    assert isinstance(response, pagers.ListTransferConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transfer_configs_async_from_dict():
    await test_list_transfer_configs_async(request_type=dict)

def test_list_transfer_configs_field_headers():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListTransferConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.return_value = datatransfer.ListTransferConfigsResponse()
        client.list_transfer_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_transfer_configs_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListTransferConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferConfigsResponse())
        await client.list_transfer_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_transfer_configs_flattened():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.return_value = datatransfer.ListTransferConfigsResponse()
        client.list_transfer_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_transfer_configs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_transfer_configs(datatransfer.ListTransferConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_transfer_configs_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.return_value = datatransfer.ListTransferConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferConfigsResponse())
        response = await client.list_transfer_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_transfer_configs_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_transfer_configs(datatransfer.ListTransferConfigsRequest(), parent='parent_value')

def test_list_transfer_configs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.side_effect = (datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig(), transfer.TransferConfig()], next_page_token='abc'), datatransfer.ListTransferConfigsResponse(transfer_configs=[], next_page_token='def'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig()], next_page_token='ghi'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_transfer_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer.TransferConfig) for i in results))

def test_list_transfer_configs_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__') as call:
        call.side_effect = (datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig(), transfer.TransferConfig()], next_page_token='abc'), datatransfer.ListTransferConfigsResponse(transfer_configs=[], next_page_token='def'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig()], next_page_token='ghi'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig()]), RuntimeError)
        pages = list(client.list_transfer_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transfer_configs_async_pager():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig(), transfer.TransferConfig()], next_page_token='abc'), datatransfer.ListTransferConfigsResponse(transfer_configs=[], next_page_token='def'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig()], next_page_token='ghi'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig()]), RuntimeError)
        async_pager = await client.list_transfer_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, transfer.TransferConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_transfer_configs_async_pages():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig(), transfer.TransferConfig()], next_page_token='abc'), datatransfer.ListTransferConfigsResponse(transfer_configs=[], next_page_token='def'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig()], next_page_token='ghi'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transfer_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.ScheduleTransferRunsRequest, dict])
def test_schedule_transfer_runs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ScheduleTransferRunsResponse()
        response = client.schedule_transfer_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ScheduleTransferRunsRequest()
    assert isinstance(response, datatransfer.ScheduleTransferRunsResponse)

def test_schedule_transfer_runs_empty_call():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        client.schedule_transfer_runs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ScheduleTransferRunsRequest()

@pytest.mark.asyncio
async def test_schedule_transfer_runs_async(transport: str='grpc_asyncio', request_type=datatransfer.ScheduleTransferRunsRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ScheduleTransferRunsResponse())
        response = await client.schedule_transfer_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ScheduleTransferRunsRequest()
    assert isinstance(response, datatransfer.ScheduleTransferRunsResponse)

@pytest.mark.asyncio
async def test_schedule_transfer_runs_async_from_dict():
    await test_schedule_transfer_runs_async(request_type=dict)

def test_schedule_transfer_runs_field_headers():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ScheduleTransferRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ScheduleTransferRunsResponse()
        client.schedule_transfer_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_schedule_transfer_runs_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ScheduleTransferRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ScheduleTransferRunsResponse())
        await client.schedule_transfer_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_schedule_transfer_runs_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ScheduleTransferRunsResponse()
        client.schedule_transfer_runs(parent='parent_value', start_time=timestamp_pb2.Timestamp(seconds=751), end_time=timestamp_pb2.Timestamp(seconds=751))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        assert TimestampRule().to_proto(args[0].start_time) == timestamp_pb2.Timestamp(seconds=751)
        assert TimestampRule().to_proto(args[0].end_time) == timestamp_pb2.Timestamp(seconds=751)

def test_schedule_transfer_runs_flattened_error():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.schedule_transfer_runs(datatransfer.ScheduleTransferRunsRequest(), parent='parent_value', start_time=timestamp_pb2.Timestamp(seconds=751), end_time=timestamp_pb2.Timestamp(seconds=751))

@pytest.mark.asyncio
async def test_schedule_transfer_runs_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.schedule_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ScheduleTransferRunsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ScheduleTransferRunsResponse())
        response = await client.schedule_transfer_runs(parent='parent_value', start_time=timestamp_pb2.Timestamp(seconds=751), end_time=timestamp_pb2.Timestamp(seconds=751))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        assert TimestampRule().to_proto(args[0].start_time) == timestamp_pb2.Timestamp(seconds=751)
        assert TimestampRule().to_proto(args[0].end_time) == timestamp_pb2.Timestamp(seconds=751)

@pytest.mark.asyncio
async def test_schedule_transfer_runs_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.schedule_transfer_runs(datatransfer.ScheduleTransferRunsRequest(), parent='parent_value', start_time=timestamp_pb2.Timestamp(seconds=751), end_time=timestamp_pb2.Timestamp(seconds=751))

@pytest.mark.parametrize('request_type', [datatransfer.StartManualTransferRunsRequest, dict])
def test_start_manual_transfer_runs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_manual_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.StartManualTransferRunsResponse()
        response = client.start_manual_transfer_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.StartManualTransferRunsRequest()
    assert isinstance(response, datatransfer.StartManualTransferRunsResponse)

def test_start_manual_transfer_runs_empty_call():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_manual_transfer_runs), '__call__') as call:
        client.start_manual_transfer_runs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.StartManualTransferRunsRequest()

@pytest.mark.asyncio
async def test_start_manual_transfer_runs_async(transport: str='grpc_asyncio', request_type=datatransfer.StartManualTransferRunsRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_manual_transfer_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.StartManualTransferRunsResponse())
        response = await client.start_manual_transfer_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.StartManualTransferRunsRequest()
    assert isinstance(response, datatransfer.StartManualTransferRunsResponse)

@pytest.mark.asyncio
async def test_start_manual_transfer_runs_async_from_dict():
    await test_start_manual_transfer_runs_async(request_type=dict)

def test_start_manual_transfer_runs_field_headers():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.StartManualTransferRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.start_manual_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.StartManualTransferRunsResponse()
        client.start_manual_transfer_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_manual_transfer_runs_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.StartManualTransferRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.start_manual_transfer_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.StartManualTransferRunsResponse())
        await client.start_manual_transfer_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [datatransfer.GetTransferRunRequest, dict])
def test_get_transfer_run(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        call.return_value = transfer.TransferRun(name='name_value', data_source_id='data_source_id_value', state=transfer.TransferState.PENDING, user_id=747, schedule='schedule_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response = client.get_transfer_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetTransferRunRequest()
    assert isinstance(response, transfer.TransferRun)
    assert response.name == 'name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.schedule == 'schedule_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_get_transfer_run_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        client.get_transfer_run()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetTransferRunRequest()

@pytest.mark.asyncio
async def test_get_transfer_run_async(transport: str='grpc_asyncio', request_type=datatransfer.GetTransferRunRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferRun(name='name_value', data_source_id='data_source_id_value', state=transfer.TransferState.PENDING, user_id=747, schedule='schedule_value', notification_pubsub_topic='notification_pubsub_topic_value'))
        response = await client.get_transfer_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.GetTransferRunRequest()
    assert isinstance(response, transfer.TransferRun)
    assert response.name == 'name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.schedule == 'schedule_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

@pytest.mark.asyncio
async def test_get_transfer_run_async_from_dict():
    await test_get_transfer_run_async(request_type=dict)

def test_get_transfer_run_field_headers():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.GetTransferRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        call.return_value = transfer.TransferRun()
        client.get_transfer_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_transfer_run_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.GetTransferRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferRun())
        await client.get_transfer_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_transfer_run_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        call.return_value = transfer.TransferRun()
        client.get_transfer_run(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_transfer_run_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_transfer_run(datatransfer.GetTransferRunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_transfer_run_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_transfer_run), '__call__') as call:
        call.return_value = transfer.TransferRun()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transfer.TransferRun())
        response = await client.get_transfer_run(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_transfer_run_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_transfer_run(datatransfer.GetTransferRunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [datatransfer.DeleteTransferRunRequest, dict])
def test_delete_transfer_run(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        call.return_value = None
        response = client.delete_transfer_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.DeleteTransferRunRequest()
    assert response is None

def test_delete_transfer_run_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        client.delete_transfer_run()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.DeleteTransferRunRequest()

@pytest.mark.asyncio
async def test_delete_transfer_run_async(transport: str='grpc_asyncio', request_type=datatransfer.DeleteTransferRunRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transfer_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.DeleteTransferRunRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_transfer_run_async_from_dict():
    await test_delete_transfer_run_async(request_type=dict)

def test_delete_transfer_run_field_headers():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.DeleteTransferRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        call.return_value = None
        client.delete_transfer_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_transfer_run_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.DeleteTransferRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_transfer_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_transfer_run_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        call.return_value = None
        client.delete_transfer_run(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_transfer_run_flattened_error():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_transfer_run(datatransfer.DeleteTransferRunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_transfer_run_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_transfer_run), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transfer_run(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_transfer_run_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_transfer_run(datatransfer.DeleteTransferRunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [datatransfer.ListTransferRunsRequest, dict])
def test_list_transfer_runs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ListTransferRunsResponse(next_page_token='next_page_token_value')
        response = client.list_transfer_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferRunsRequest()
    assert isinstance(response, pagers.ListTransferRunsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_runs_empty_call():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        client.list_transfer_runs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferRunsRequest()

@pytest.mark.asyncio
async def test_list_transfer_runs_async(transport: str='grpc_asyncio', request_type=datatransfer.ListTransferRunsRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferRunsResponse(next_page_token='next_page_token_value'))
        response = await client.list_transfer_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferRunsRequest()
    assert isinstance(response, pagers.ListTransferRunsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transfer_runs_async_from_dict():
    await test_list_transfer_runs_async(request_type=dict)

def test_list_transfer_runs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListTransferRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ListTransferRunsResponse()
        client.list_transfer_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_transfer_runs_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListTransferRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferRunsResponse())
        await client.list_transfer_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_transfer_runs_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ListTransferRunsResponse()
        client.list_transfer_runs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_transfer_runs_flattened_error():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_transfer_runs(datatransfer.ListTransferRunsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_transfer_runs_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.return_value = datatransfer.ListTransferRunsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferRunsResponse())
        response = await client.list_transfer_runs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_transfer_runs_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_transfer_runs(datatransfer.ListTransferRunsRequest(), parent='parent_value')

def test_list_transfer_runs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.side_effect = (datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun(), transfer.TransferRun()], next_page_token='abc'), datatransfer.ListTransferRunsResponse(transfer_runs=[], next_page_token='def'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun()], next_page_token='ghi'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_transfer_runs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer.TransferRun) for i in results))

def test_list_transfer_runs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__') as call:
        call.side_effect = (datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun(), transfer.TransferRun()], next_page_token='abc'), datatransfer.ListTransferRunsResponse(transfer_runs=[], next_page_token='def'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun()], next_page_token='ghi'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun()]), RuntimeError)
        pages = list(client.list_transfer_runs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transfer_runs_async_pager():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun(), transfer.TransferRun()], next_page_token='abc'), datatransfer.ListTransferRunsResponse(transfer_runs=[], next_page_token='def'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun()], next_page_token='ghi'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun()]), RuntimeError)
        async_pager = await client.list_transfer_runs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, transfer.TransferRun) for i in responses))

@pytest.mark.asyncio
async def test_list_transfer_runs_async_pages():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_runs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun(), transfer.TransferRun()], next_page_token='abc'), datatransfer.ListTransferRunsResponse(transfer_runs=[], next_page_token='def'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun()], next_page_token='ghi'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transfer_runs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.ListTransferLogsRequest, dict])
def test_list_transfer_logs(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.return_value = datatransfer.ListTransferLogsResponse(next_page_token='next_page_token_value')
        response = client.list_transfer_logs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferLogsRequest()
    assert isinstance(response, pagers.ListTransferLogsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_logs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        client.list_transfer_logs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferLogsRequest()

@pytest.mark.asyncio
async def test_list_transfer_logs_async(transport: str='grpc_asyncio', request_type=datatransfer.ListTransferLogsRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferLogsResponse(next_page_token='next_page_token_value'))
        response = await client.list_transfer_logs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.ListTransferLogsRequest()
    assert isinstance(response, pagers.ListTransferLogsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transfer_logs_async_from_dict():
    await test_list_transfer_logs_async(request_type=dict)

def test_list_transfer_logs_field_headers():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListTransferLogsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.return_value = datatransfer.ListTransferLogsResponse()
        client.list_transfer_logs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_transfer_logs_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.ListTransferLogsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferLogsResponse())
        await client.list_transfer_logs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_transfer_logs_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.return_value = datatransfer.ListTransferLogsResponse()
        client.list_transfer_logs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_transfer_logs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_transfer_logs(datatransfer.ListTransferLogsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_transfer_logs_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.return_value = datatransfer.ListTransferLogsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.ListTransferLogsResponse())
        response = await client.list_transfer_logs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_transfer_logs_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_transfer_logs(datatransfer.ListTransferLogsRequest(), parent='parent_value')

def test_list_transfer_logs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.side_effect = (datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage(), transfer.TransferMessage()], next_page_token='abc'), datatransfer.ListTransferLogsResponse(transfer_messages=[], next_page_token='def'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage()], next_page_token='ghi'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_transfer_logs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer.TransferMessage) for i in results))

def test_list_transfer_logs_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__') as call:
        call.side_effect = (datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage(), transfer.TransferMessage()], next_page_token='abc'), datatransfer.ListTransferLogsResponse(transfer_messages=[], next_page_token='def'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage()], next_page_token='ghi'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage()]), RuntimeError)
        pages = list(client.list_transfer_logs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transfer_logs_async_pager():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage(), transfer.TransferMessage()], next_page_token='abc'), datatransfer.ListTransferLogsResponse(transfer_messages=[], next_page_token='def'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage()], next_page_token='ghi'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage()]), RuntimeError)
        async_pager = await client.list_transfer_logs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, transfer.TransferMessage) for i in responses))

@pytest.mark.asyncio
async def test_list_transfer_logs_async_pages():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transfer_logs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage(), transfer.TransferMessage()], next_page_token='abc'), datatransfer.ListTransferLogsResponse(transfer_messages=[], next_page_token='def'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage()], next_page_token='ghi'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transfer_logs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.CheckValidCredsRequest, dict])
def test_check_valid_creds(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        call.return_value = datatransfer.CheckValidCredsResponse(has_valid_creds=True)
        response = client.check_valid_creds(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.CheckValidCredsRequest()
    assert isinstance(response, datatransfer.CheckValidCredsResponse)
    assert response.has_valid_creds is True

def test_check_valid_creds_empty_call():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        client.check_valid_creds()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.CheckValidCredsRequest()

@pytest.mark.asyncio
async def test_check_valid_creds_async(transport: str='grpc_asyncio', request_type=datatransfer.CheckValidCredsRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.CheckValidCredsResponse(has_valid_creds=True))
        response = await client.check_valid_creds(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.CheckValidCredsRequest()
    assert isinstance(response, datatransfer.CheckValidCredsResponse)
    assert response.has_valid_creds is True

@pytest.mark.asyncio
async def test_check_valid_creds_async_from_dict():
    await test_check_valid_creds_async(request_type=dict)

def test_check_valid_creds_field_headers():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.CheckValidCredsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        call.return_value = datatransfer.CheckValidCredsResponse()
        client.check_valid_creds(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_check_valid_creds_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.CheckValidCredsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.CheckValidCredsResponse())
        await client.check_valid_creds(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_check_valid_creds_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        call.return_value = datatransfer.CheckValidCredsResponse()
        client.check_valid_creds(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_check_valid_creds_flattened_error():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.check_valid_creds(datatransfer.CheckValidCredsRequest(), name='name_value')

@pytest.mark.asyncio
async def test_check_valid_creds_flattened_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.check_valid_creds), '__call__') as call:
        call.return_value = datatransfer.CheckValidCredsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(datatransfer.CheckValidCredsResponse())
        response = await client.check_valid_creds(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_check_valid_creds_flattened_error_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.check_valid_creds(datatransfer.CheckValidCredsRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [datatransfer.EnrollDataSourcesRequest, dict])
def test_enroll_data_sources(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enroll_data_sources), '__call__') as call:
        call.return_value = None
        response = client.enroll_data_sources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.EnrollDataSourcesRequest()
    assert response is None

def test_enroll_data_sources_empty_call():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.enroll_data_sources), '__call__') as call:
        client.enroll_data_sources()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.EnrollDataSourcesRequest()

@pytest.mark.asyncio
async def test_enroll_data_sources_async(transport: str='grpc_asyncio', request_type=datatransfer.EnrollDataSourcesRequest):
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enroll_data_sources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.enroll_data_sources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == datatransfer.EnrollDataSourcesRequest()
    assert response is None

@pytest.mark.asyncio
async def test_enroll_data_sources_async_from_dict():
    await test_enroll_data_sources_async(request_type=dict)

def test_enroll_data_sources_field_headers():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.EnrollDataSourcesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enroll_data_sources), '__call__') as call:
        call.return_value = None
        client.enroll_data_sources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_enroll_data_sources_field_headers_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = datatransfer.EnrollDataSourcesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enroll_data_sources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.enroll_data_sources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [datatransfer.GetDataSourceRequest, dict])
def test_get_data_source_rest(request_type):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dataSources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.DataSource(name='name_value', data_source_id='data_source_id_value', display_name='display_name_value', description='description_value', client_id='client_id_value', scopes=['scopes_value'], transfer_type=transfer.TransferType.BATCH, supports_multiple_transfers=True, update_deadline_seconds=2406, default_schedule='default_schedule_value', supports_custom_schedule=True, help_url='help_url_value', authorization_type=datatransfer.DataSource.AuthorizationType.AUTHORIZATION_CODE, data_refresh_type=datatransfer.DataSource.DataRefreshType.SLIDING_WINDOW, default_data_refresh_window_days=3379, manual_runs_disabled=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.DataSource.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_data_source(request)
    assert isinstance(response, datatransfer.DataSource)
    assert response.name == 'name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.client_id == 'client_id_value'
    assert response.scopes == ['scopes_value']
    assert response.transfer_type == transfer.TransferType.BATCH
    assert response.supports_multiple_transfers is True
    assert response.update_deadline_seconds == 2406
    assert response.default_schedule == 'default_schedule_value'
    assert response.supports_custom_schedule is True
    assert response.help_url == 'help_url_value'
    assert response.authorization_type == datatransfer.DataSource.AuthorizationType.AUTHORIZATION_CODE
    assert response.data_refresh_type == datatransfer.DataSource.DataRefreshType.SLIDING_WINDOW
    assert response.default_data_refresh_window_days == 3379
    assert response.manual_runs_disabled is True

def test_get_data_source_rest_required_fields(request_type=datatransfer.GetDataSourceRequest):
    if False:
        return 10
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_data_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_data_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.DataSource()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.DataSource.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_data_source(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_data_source_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_data_source._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_data_source_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_get_data_source') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_get_data_source') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.GetDataSourceRequest.pb(datatransfer.GetDataSourceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.DataSource.to_json(datatransfer.DataSource())
        request = datatransfer.GetDataSourceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.DataSource()
        client.get_data_source(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_data_source_rest_bad_request(transport: str='rest', request_type=datatransfer.GetDataSourceRequest):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dataSources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_data_source(request)

def test_get_data_source_rest_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.DataSource()
        sample_request = {'name': 'projects/sample1/locations/sample2/dataSources/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.DataSource.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_data_source(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/dataSources/*}' % client.transport._host, args[1])

def test_get_data_source_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_data_source(datatransfer.GetDataSourceRequest(), name='name_value')

def test_get_data_source_rest_error():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.ListDataSourcesRequest, dict])
def test_list_data_sources_rest(request_type):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListDataSourcesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListDataSourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_data_sources(request)
    assert isinstance(response, pagers.ListDataSourcesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_data_sources_rest_required_fields(request_type=datatransfer.ListDataSourcesRequest):
    if False:
        return 10
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_data_sources._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_data_sources._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.ListDataSourcesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.ListDataSourcesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_data_sources(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_data_sources_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_data_sources._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_data_sources_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_list_data_sources') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_list_data_sources') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.ListDataSourcesRequest.pb(datatransfer.ListDataSourcesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.ListDataSourcesResponse.to_json(datatransfer.ListDataSourcesResponse())
        request = datatransfer.ListDataSourcesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.ListDataSourcesResponse()
        client.list_data_sources(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_data_sources_rest_bad_request(transport: str='rest', request_type=datatransfer.ListDataSourcesRequest):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_data_sources(request)

def test_list_data_sources_rest_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListDataSourcesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListDataSourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_data_sources(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/dataSources' % client.transport._host, args[1])

def test_list_data_sources_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_data_sources(datatransfer.ListDataSourcesRequest(), parent='parent_value')

def test_list_data_sources_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource(), datatransfer.DataSource()], next_page_token='abc'), datatransfer.ListDataSourcesResponse(data_sources=[], next_page_token='def'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource()], next_page_token='ghi'), datatransfer.ListDataSourcesResponse(data_sources=[datatransfer.DataSource(), datatransfer.DataSource()]))
        response = response + response
        response = tuple((datatransfer.ListDataSourcesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_data_sources(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, datatransfer.DataSource) for i in results))
        pages = list(client.list_data_sources(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.CreateTransferConfigRequest, dict])
def test_create_transfer_config_rest(request_type):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['transfer_config'] = {'name': 'name_value', 'destination_dataset_id': 'destination_dataset_id_value', 'display_name': 'display_name_value', 'data_source_id': 'data_source_id_value', 'params': {'fields': {}}, 'schedule': 'schedule_value', 'schedule_options': {'disable_auto_scheduling': True, 'start_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}}, 'data_refresh_window_days': 2543, 'disabled': True, 'update_time': {}, 'next_run_time': {}, 'state': 2, 'user_id': 747, 'dataset_region': 'dataset_region_value', 'notification_pubsub_topic': 'notification_pubsub_topic_value', 'email_preferences': {'enable_failure_email': True}, 'owner_info': {'email': 'email_value'}, 'encryption_configuration': {'kms_key_name': {'value': 'value_value'}}}
    test_field = datatransfer.CreateTransferConfigRequest.meta.fields['transfer_config']

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
    for (field, value) in request_init['transfer_config'].items():
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
                for i in range(0, len(request_init['transfer_config'][field])):
                    del request_init['transfer_config'][field][i][subfield]
            else:
                del request_init['transfer_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_transfer_config(request)
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_create_transfer_config_rest_required_fields(request_type=datatransfer.CreateTransferConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_transfer_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_transfer_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('authorization_code', 'service_account_name', 'version_info'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer.TransferConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer.TransferConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_transfer_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_transfer_config_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_transfer_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('authorizationCode', 'serviceAccountName', 'versionInfo')) & set(('parent', 'transferConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_transfer_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_create_transfer_config') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_create_transfer_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.CreateTransferConfigRequest.pb(datatransfer.CreateTransferConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer.TransferConfig.to_json(transfer.TransferConfig())
        request = datatransfer.CreateTransferConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer.TransferConfig()
        client.create_transfer_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_transfer_config_rest_bad_request(transport: str='rest', request_type=datatransfer.CreateTransferConfigRequest):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_transfer_config(request)

def test_create_transfer_config_rest_flattened():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferConfig()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', transfer_config=transfer.TransferConfig(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_transfer_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/transferConfigs' % client.transport._host, args[1])

def test_create_transfer_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_transfer_config(datatransfer.CreateTransferConfigRequest(), parent='parent_value', transfer_config=transfer.TransferConfig(name='name_value'))

def test_create_transfer_config_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.UpdateTransferConfigRequest, dict])
def test_update_transfer_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'transfer_config': {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}}
    request_init['transfer_config'] = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3', 'destination_dataset_id': 'destination_dataset_id_value', 'display_name': 'display_name_value', 'data_source_id': 'data_source_id_value', 'params': {'fields': {}}, 'schedule': 'schedule_value', 'schedule_options': {'disable_auto_scheduling': True, 'start_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}}, 'data_refresh_window_days': 2543, 'disabled': True, 'update_time': {}, 'next_run_time': {}, 'state': 2, 'user_id': 747, 'dataset_region': 'dataset_region_value', 'notification_pubsub_topic': 'notification_pubsub_topic_value', 'email_preferences': {'enable_failure_email': True}, 'owner_info': {'email': 'email_value'}, 'encryption_configuration': {'kms_key_name': {'value': 'value_value'}}}
    test_field = datatransfer.UpdateTransferConfigRequest.meta.fields['transfer_config']

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
    for (field, value) in request_init['transfer_config'].items():
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
                for i in range(0, len(request_init['transfer_config'][field])):
                    del request_init['transfer_config'][field][i][subfield]
            else:
                del request_init['transfer_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_transfer_config(request)
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_update_transfer_config_rest_required_fields(request_type=datatransfer.UpdateTransferConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_transfer_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_transfer_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('authorization_code', 'service_account_name', 'update_mask', 'version_info'))
    jsonified_request.update(unset_fields)
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer.TransferConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer.TransferConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_transfer_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_transfer_config_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_transfer_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('authorizationCode', 'serviceAccountName', 'updateMask', 'versionInfo')) & set(('transferConfig', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_transfer_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_update_transfer_config') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_update_transfer_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.UpdateTransferConfigRequest.pb(datatransfer.UpdateTransferConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer.TransferConfig.to_json(transfer.TransferConfig())
        request = datatransfer.UpdateTransferConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer.TransferConfig()
        client.update_transfer_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_transfer_config_rest_bad_request(transport: str='rest', request_type=datatransfer.UpdateTransferConfigRequest):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'transfer_config': {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_transfer_config(request)

def test_update_transfer_config_rest_flattened():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferConfig()
        sample_request = {'transfer_config': {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}}
        mock_args = dict(transfer_config=transfer.TransferConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_transfer_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{transfer_config.name=projects/*/locations/*/transferConfigs/*}' % client.transport._host, args[1])

def test_update_transfer_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_transfer_config(datatransfer.UpdateTransferConfigRequest(), transfer_config=transfer.TransferConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_transfer_config_rest_error():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.DeleteTransferConfigRequest, dict])
def test_delete_transfer_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_transfer_config(request)
    assert response is None

def test_delete_transfer_config_rest_required_fields(request_type=datatransfer.DeleteTransferConfigRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transfer_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transfer_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_transfer_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_transfer_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_transfer_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_transfer_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_delete_transfer_config') as pre:
        pre.assert_not_called()
        pb_message = datatransfer.DeleteTransferConfigRequest.pb(datatransfer.DeleteTransferConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = datatransfer.DeleteTransferConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_transfer_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_transfer_config_rest_bad_request(transport: str='rest', request_type=datatransfer.DeleteTransferConfigRequest):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_transfer_config(request)

def test_delete_transfer_config_rest_flattened():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_transfer_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/transferConfigs/*}' % client.transport._host, args[1])

def test_delete_transfer_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_transfer_config(datatransfer.DeleteTransferConfigRequest(), name='name_value')

def test_delete_transfer_config_rest_error():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.GetTransferConfigRequest, dict])
def test_get_transfer_config_rest(request_type):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferConfig(name='name_value', display_name='display_name_value', data_source_id='data_source_id_value', schedule='schedule_value', data_refresh_window_days=2543, disabled=True, state=transfer.TransferState.PENDING, user_id=747, dataset_region='dataset_region_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_transfer_config(request)
    assert isinstance(response, transfer.TransferConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.schedule == 'schedule_value'
    assert response.data_refresh_window_days == 2543
    assert response.disabled is True
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.dataset_region == 'dataset_region_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_get_transfer_config_rest_required_fields(request_type=datatransfer.GetTransferConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transfer_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transfer_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer.TransferConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer.TransferConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_transfer_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_transfer_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_transfer_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_transfer_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_get_transfer_config') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_get_transfer_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.GetTransferConfigRequest.pb(datatransfer.GetTransferConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer.TransferConfig.to_json(transfer.TransferConfig())
        request = datatransfer.GetTransferConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer.TransferConfig()
        client.get_transfer_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_transfer_config_rest_bad_request(transport: str='rest', request_type=datatransfer.GetTransferConfigRequest):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_transfer_config(request)

def test_get_transfer_config_rest_flattened():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_transfer_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/transferConfigs/*}' % client.transport._host, args[1])

def test_get_transfer_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_transfer_config(datatransfer.GetTransferConfigRequest(), name='name_value')

def test_get_transfer_config_rest_error():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.ListTransferConfigsRequest, dict])
def test_list_transfer_configs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListTransferConfigsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListTransferConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_transfer_configs(request)
    assert isinstance(response, pagers.ListTransferConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_configs_rest_required_fields(request_type=datatransfer.ListTransferConfigsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('data_source_ids', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.ListTransferConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.ListTransferConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_transfer_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_transfer_configs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_transfer_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('dataSourceIds', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_transfer_configs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_list_transfer_configs') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_list_transfer_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.ListTransferConfigsRequest.pb(datatransfer.ListTransferConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.ListTransferConfigsResponse.to_json(datatransfer.ListTransferConfigsResponse())
        request = datatransfer.ListTransferConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.ListTransferConfigsResponse()
        client.list_transfer_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_transfer_configs_rest_bad_request(transport: str='rest', request_type=datatransfer.ListTransferConfigsRequest):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_transfer_configs(request)

def test_list_transfer_configs_rest_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListTransferConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListTransferConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_transfer_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/transferConfigs' % client.transport._host, args[1])

def test_list_transfer_configs_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_transfer_configs(datatransfer.ListTransferConfigsRequest(), parent='parent_value')

def test_list_transfer_configs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig(), transfer.TransferConfig()], next_page_token='abc'), datatransfer.ListTransferConfigsResponse(transfer_configs=[], next_page_token='def'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig()], next_page_token='ghi'), datatransfer.ListTransferConfigsResponse(transfer_configs=[transfer.TransferConfig(), transfer.TransferConfig()]))
        response = response + response
        response = tuple((datatransfer.ListTransferConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_transfer_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer.TransferConfig) for i in results))
        pages = list(client.list_transfer_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.ScheduleTransferRunsRequest, dict])
def test_schedule_transfer_runs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ScheduleTransferRunsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ScheduleTransferRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.schedule_transfer_runs(request)
    assert isinstance(response, datatransfer.ScheduleTransferRunsResponse)

def test_schedule_transfer_runs_rest_required_fields(request_type=datatransfer.ScheduleTransferRunsRequest):
    if False:
        return 10
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).schedule_transfer_runs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).schedule_transfer_runs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.ScheduleTransferRunsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.ScheduleTransferRunsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.schedule_transfer_runs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_schedule_transfer_runs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.schedule_transfer_runs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'startTime', 'endTime'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_schedule_transfer_runs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_schedule_transfer_runs') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_schedule_transfer_runs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.ScheduleTransferRunsRequest.pb(datatransfer.ScheduleTransferRunsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.ScheduleTransferRunsResponse.to_json(datatransfer.ScheduleTransferRunsResponse())
        request = datatransfer.ScheduleTransferRunsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.ScheduleTransferRunsResponse()
        client.schedule_transfer_runs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_schedule_transfer_runs_rest_bad_request(transport: str='rest', request_type=datatransfer.ScheduleTransferRunsRequest):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.schedule_transfer_runs(request)

def test_schedule_transfer_runs_rest_flattened():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ScheduleTransferRunsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
        mock_args = dict(parent='parent_value', start_time=timestamp_pb2.Timestamp(seconds=751), end_time=timestamp_pb2.Timestamp(seconds=751))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ScheduleTransferRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.schedule_transfer_runs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/transferConfigs/*}:scheduleRuns' % client.transport._host, args[1])

def test_schedule_transfer_runs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.schedule_transfer_runs(datatransfer.ScheduleTransferRunsRequest(), parent='parent_value', start_time=timestamp_pb2.Timestamp(seconds=751), end_time=timestamp_pb2.Timestamp(seconds=751))

def test_schedule_transfer_runs_rest_error():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.StartManualTransferRunsRequest, dict])
def test_start_manual_transfer_runs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.StartManualTransferRunsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.StartManualTransferRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_manual_transfer_runs(request)
    assert isinstance(response, datatransfer.StartManualTransferRunsResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_manual_transfer_runs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_start_manual_transfer_runs') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_start_manual_transfer_runs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.StartManualTransferRunsRequest.pb(datatransfer.StartManualTransferRunsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.StartManualTransferRunsResponse.to_json(datatransfer.StartManualTransferRunsResponse())
        request = datatransfer.StartManualTransferRunsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.StartManualTransferRunsResponse()
        client.start_manual_transfer_runs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_manual_transfer_runs_rest_bad_request(transport: str='rest', request_type=datatransfer.StartManualTransferRunsRequest):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_manual_transfer_runs(request)

def test_start_manual_transfer_runs_rest_error():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.GetTransferRunRequest, dict])
def test_get_transfer_run_rest(request_type):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferRun(name='name_value', data_source_id='data_source_id_value', state=transfer.TransferState.PENDING, user_id=747, schedule='schedule_value', notification_pubsub_topic='notification_pubsub_topic_value', destination_dataset_id='destination_dataset_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferRun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_transfer_run(request)
    assert isinstance(response, transfer.TransferRun)
    assert response.name == 'name_value'
    assert response.data_source_id == 'data_source_id_value'
    assert response.state == transfer.TransferState.PENDING
    assert response.user_id == 747
    assert response.schedule == 'schedule_value'
    assert response.notification_pubsub_topic == 'notification_pubsub_topic_value'

def test_get_transfer_run_rest_required_fields(request_type=datatransfer.GetTransferRunRequest):
    if False:
        return 10
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transfer_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transfer_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transfer.TransferRun()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transfer.TransferRun.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_transfer_run(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_transfer_run_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_transfer_run._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_transfer_run_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_get_transfer_run') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_get_transfer_run') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.GetTransferRunRequest.pb(datatransfer.GetTransferRunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transfer.TransferRun.to_json(transfer.TransferRun())
        request = datatransfer.GetTransferRunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transfer.TransferRun()
        client.get_transfer_run(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_transfer_run_rest_bad_request(transport: str='rest', request_type=datatransfer.GetTransferRunRequest):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_transfer_run(request)

def test_get_transfer_run_rest_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transfer.TransferRun()
        sample_request = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transfer.TransferRun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_transfer_run(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/transferConfigs/*/runs/*}' % client.transport._host, args[1])

def test_get_transfer_run_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_transfer_run(datatransfer.GetTransferRunRequest(), name='name_value')

def test_get_transfer_run_rest_error():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.DeleteTransferRunRequest, dict])
def test_delete_transfer_run_rest(request_type):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_transfer_run(request)
    assert response is None

def test_delete_transfer_run_rest_required_fields(request_type=datatransfer.DeleteTransferRunRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transfer_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transfer_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_transfer_run(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_transfer_run_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_transfer_run._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_transfer_run_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_delete_transfer_run') as pre:
        pre.assert_not_called()
        pb_message = datatransfer.DeleteTransferRunRequest.pb(datatransfer.DeleteTransferRunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = datatransfer.DeleteTransferRunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_transfer_run(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_transfer_run_rest_bad_request(transport: str='rest', request_type=datatransfer.DeleteTransferRunRequest):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_transfer_run(request)

def test_delete_transfer_run_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_transfer_run(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/transferConfigs/*/runs/*}' % client.transport._host, args[1])

def test_delete_transfer_run_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_transfer_run(datatransfer.DeleteTransferRunRequest(), name='name_value')

def test_delete_transfer_run_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.ListTransferRunsRequest, dict])
def test_list_transfer_runs_rest(request_type):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListTransferRunsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListTransferRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_transfer_runs(request)
    assert isinstance(response, pagers.ListTransferRunsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_runs_rest_required_fields(request_type=datatransfer.ListTransferRunsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_runs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_runs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'run_attempt', 'states'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.ListTransferRunsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.ListTransferRunsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_transfer_runs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_transfer_runs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_transfer_runs._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'runAttempt', 'states')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_transfer_runs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_list_transfer_runs') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_list_transfer_runs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.ListTransferRunsRequest.pb(datatransfer.ListTransferRunsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.ListTransferRunsResponse.to_json(datatransfer.ListTransferRunsResponse())
        request = datatransfer.ListTransferRunsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.ListTransferRunsResponse()
        client.list_transfer_runs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_transfer_runs_rest_bad_request(transport: str='rest', request_type=datatransfer.ListTransferRunsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_transfer_runs(request)

def test_list_transfer_runs_rest_flattened():
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListTransferRunsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListTransferRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_transfer_runs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/transferConfigs/*}/runs' % client.transport._host, args[1])

def test_list_transfer_runs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_transfer_runs(datatransfer.ListTransferRunsRequest(), parent='parent_value')

def test_list_transfer_runs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun(), transfer.TransferRun()], next_page_token='abc'), datatransfer.ListTransferRunsResponse(transfer_runs=[], next_page_token='def'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun()], next_page_token='ghi'), datatransfer.ListTransferRunsResponse(transfer_runs=[transfer.TransferRun(), transfer.TransferRun()]))
        response = response + response
        response = tuple((datatransfer.ListTransferRunsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3'}
        pager = client.list_transfer_runs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer.TransferRun) for i in results))
        pages = list(client.list_transfer_runs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.ListTransferLogsRequest, dict])
def test_list_transfer_logs_rest(request_type):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListTransferLogsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListTransferLogsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_transfer_logs(request)
    assert isinstance(response, pagers.ListTransferLogsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transfer_logs_rest_required_fields(request_type=datatransfer.ListTransferLogsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_logs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transfer_logs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('message_types', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.ListTransferLogsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.ListTransferLogsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_transfer_logs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_transfer_logs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_transfer_logs._get_unset_required_fields({})
    assert set(unset_fields) == set(('messageTypes', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_transfer_logs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_list_transfer_logs') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_list_transfer_logs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.ListTransferLogsRequest.pb(datatransfer.ListTransferLogsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.ListTransferLogsResponse.to_json(datatransfer.ListTransferLogsResponse())
        request = datatransfer.ListTransferLogsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.ListTransferLogsResponse()
        client.list_transfer_logs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_transfer_logs_rest_bad_request(transport: str='rest', request_type=datatransfer.ListTransferLogsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_transfer_logs(request)

def test_list_transfer_logs_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.ListTransferLogsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.ListTransferLogsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_transfer_logs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/transferConfigs/*/runs/*}/transferLogs' % client.transport._host, args[1])

def test_list_transfer_logs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_transfer_logs(datatransfer.ListTransferLogsRequest(), parent='parent_value')

def test_list_transfer_logs_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage(), transfer.TransferMessage()], next_page_token='abc'), datatransfer.ListTransferLogsResponse(transfer_messages=[], next_page_token='def'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage()], next_page_token='ghi'), datatransfer.ListTransferLogsResponse(transfer_messages=[transfer.TransferMessage(), transfer.TransferMessage()]))
        response = response + response
        response = tuple((datatransfer.ListTransferLogsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/transferConfigs/sample3/runs/sample4'}
        pager = client.list_transfer_logs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transfer.TransferMessage) for i in results))
        pages = list(client.list_transfer_logs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [datatransfer.CheckValidCredsRequest, dict])
def test_check_valid_creds_rest(request_type):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dataSources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.CheckValidCredsResponse(has_valid_creds=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.CheckValidCredsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.check_valid_creds(request)
    assert isinstance(response, datatransfer.CheckValidCredsResponse)
    assert response.has_valid_creds is True

def test_check_valid_creds_rest_required_fields(request_type=datatransfer.CheckValidCredsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DataTransferServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).check_valid_creds._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).check_valid_creds._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = datatransfer.CheckValidCredsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = datatransfer.CheckValidCredsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.check_valid_creds(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_check_valid_creds_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.check_valid_creds._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_check_valid_creds_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'post_check_valid_creds') as post, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_check_valid_creds') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = datatransfer.CheckValidCredsRequest.pb(datatransfer.CheckValidCredsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = datatransfer.CheckValidCredsResponse.to_json(datatransfer.CheckValidCredsResponse())
        request = datatransfer.CheckValidCredsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = datatransfer.CheckValidCredsResponse()
        client.check_valid_creds(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_check_valid_creds_rest_bad_request(transport: str='rest', request_type=datatransfer.CheckValidCredsRequest):
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dataSources/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.check_valid_creds(request)

def test_check_valid_creds_rest_flattened():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = datatransfer.CheckValidCredsResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/dataSources/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = datatransfer.CheckValidCredsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.check_valid_creds(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/dataSources/*}:checkValidCreds' % client.transport._host, args[1])

def test_check_valid_creds_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.check_valid_creds(datatransfer.CheckValidCredsRequest(), name='name_value')

def test_check_valid_creds_rest_error():
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [datatransfer.EnrollDataSourcesRequest, dict])
def test_enroll_data_sources_rest(request_type):
    if False:
        while True:
            i = 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.enroll_data_sources(request)
    assert response is None

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_enroll_data_sources_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DataTransferServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DataTransferServiceRestInterceptor())
    client = DataTransferServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DataTransferServiceRestInterceptor, 'pre_enroll_data_sources') as pre:
        pre.assert_not_called()
        pb_message = datatransfer.EnrollDataSourcesRequest.pb(datatransfer.EnrollDataSourcesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = datatransfer.EnrollDataSourcesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.enroll_data_sources(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_enroll_data_sources_rest_bad_request(transport: str='rest', request_type=datatransfer.EnrollDataSourcesRequest):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.enroll_data_sources(request)

def test_enroll_data_sources_rest_error():
    if False:
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DataTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DataTransferServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DataTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DataTransferServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DataTransferServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DataTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DataTransferServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DataTransferServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.DataTransferServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DataTransferServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceGrpcAsyncIOTransport, transports.DataTransferServiceRestTransport])
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
        print('Hello World!')
    transport = DataTransferServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DataTransferServiceGrpcTransport)

def test_data_transfer_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DataTransferServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_data_transfer_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.bigquery_datatransfer_v1.services.data_transfer_service.transports.DataTransferServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DataTransferServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_data_source', 'list_data_sources', 'create_transfer_config', 'update_transfer_config', 'delete_transfer_config', 'get_transfer_config', 'list_transfer_configs', 'schedule_transfer_runs', 'start_manual_transfer_runs', 'get_transfer_run', 'delete_transfer_run', 'list_transfer_runs', 'list_transfer_logs', 'check_valid_creds', 'enroll_data_sources', 'get_location', 'list_locations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_data_transfer_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.bigquery_datatransfer_v1.services.data_transfer_service.transports.DataTransferServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DataTransferServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_data_transfer_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.bigquery_datatransfer_v1.services.data_transfer_service.transports.DataTransferServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DataTransferServiceTransport()
        adc.assert_called_once()

def test_data_transfer_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DataTransferServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceGrpcAsyncIOTransport])
def test_data_transfer_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceGrpcAsyncIOTransport, transports.DataTransferServiceRestTransport])
def test_data_transfer_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DataTransferServiceGrpcTransport, grpc_helpers), (transports.DataTransferServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_data_transfer_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('bigquerydatatransfer.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='bigquerydatatransfer.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceGrpcAsyncIOTransport])
def test_data_transfer_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_data_transfer_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.DataTransferServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_data_transfer_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='bigquerydatatransfer.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('bigquerydatatransfer.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigquerydatatransfer.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_data_transfer_service_host_with_port(transport_name):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='bigquerydatatransfer.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('bigquerydatatransfer.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigquerydatatransfer.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_data_transfer_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = DataTransferServiceClient(credentials=creds1, transport=transport_name)
    client2 = DataTransferServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_data_source._session
    session2 = client2.transport.get_data_source._session
    assert session1 != session2
    session1 = client1.transport.list_data_sources._session
    session2 = client2.transport.list_data_sources._session
    assert session1 != session2
    session1 = client1.transport.create_transfer_config._session
    session2 = client2.transport.create_transfer_config._session
    assert session1 != session2
    session1 = client1.transport.update_transfer_config._session
    session2 = client2.transport.update_transfer_config._session
    assert session1 != session2
    session1 = client1.transport.delete_transfer_config._session
    session2 = client2.transport.delete_transfer_config._session
    assert session1 != session2
    session1 = client1.transport.get_transfer_config._session
    session2 = client2.transport.get_transfer_config._session
    assert session1 != session2
    session1 = client1.transport.list_transfer_configs._session
    session2 = client2.transport.list_transfer_configs._session
    assert session1 != session2
    session1 = client1.transport.schedule_transfer_runs._session
    session2 = client2.transport.schedule_transfer_runs._session
    assert session1 != session2
    session1 = client1.transport.start_manual_transfer_runs._session
    session2 = client2.transport.start_manual_transfer_runs._session
    assert session1 != session2
    session1 = client1.transport.get_transfer_run._session
    session2 = client2.transport.get_transfer_run._session
    assert session1 != session2
    session1 = client1.transport.delete_transfer_run._session
    session2 = client2.transport.delete_transfer_run._session
    assert session1 != session2
    session1 = client1.transport.list_transfer_runs._session
    session2 = client2.transport.list_transfer_runs._session
    assert session1 != session2
    session1 = client1.transport.list_transfer_logs._session
    session2 = client2.transport.list_transfer_logs._session
    assert session1 != session2
    session1 = client1.transport.check_valid_creds._session
    session2 = client2.transport.check_valid_creds._session
    assert session1 != session2
    session1 = client1.transport.enroll_data_sources._session
    session2 = client2.transport.enroll_data_sources._session
    assert session1 != session2

def test_data_transfer_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DataTransferServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_data_transfer_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DataTransferServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceGrpcAsyncIOTransport])
def test_data_transfer_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.DataTransferServiceGrpcTransport, transports.DataTransferServiceGrpcAsyncIOTransport])
def test_data_transfer_service_transport_channel_mtls_with_adc(transport_class):
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

def test_data_source_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    data_source = 'clam'
    expected = 'projects/{project}/dataSources/{data_source}'.format(project=project, data_source=data_source)
    actual = DataTransferServiceClient.data_source_path(project, data_source)
    assert expected == actual

def test_parse_data_source_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'data_source': 'octopus'}
    path = DataTransferServiceClient.data_source_path(**expected)
    actual = DataTransferServiceClient.parse_data_source_path(path)
    assert expected == actual

def test_run_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    transfer_config = 'nudibranch'
    run = 'cuttlefish'
    expected = 'projects/{project}/transferConfigs/{transfer_config}/runs/{run}'.format(project=project, transfer_config=transfer_config, run=run)
    actual = DataTransferServiceClient.run_path(project, transfer_config, run)
    assert expected == actual

def test_parse_run_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel', 'transfer_config': 'winkle', 'run': 'nautilus'}
    path = DataTransferServiceClient.run_path(**expected)
    actual = DataTransferServiceClient.parse_run_path(path)
    assert expected == actual

def test_transfer_config_path():
    if False:
        return 10
    project = 'scallop'
    transfer_config = 'abalone'
    expected = 'projects/{project}/transferConfigs/{transfer_config}'.format(project=project, transfer_config=transfer_config)
    actual = DataTransferServiceClient.transfer_config_path(project, transfer_config)
    assert expected == actual

def test_parse_transfer_config_path():
    if False:
        return 10
    expected = {'project': 'squid', 'transfer_config': 'clam'}
    path = DataTransferServiceClient.transfer_config_path(**expected)
    actual = DataTransferServiceClient.parse_transfer_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DataTransferServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'octopus'}
    path = DataTransferServiceClient.common_billing_account_path(**expected)
    actual = DataTransferServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DataTransferServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nudibranch'}
    path = DataTransferServiceClient.common_folder_path(**expected)
    actual = DataTransferServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DataTransferServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'mussel'}
    path = DataTransferServiceClient.common_organization_path(**expected)
    actual = DataTransferServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = DataTransferServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus'}
    path = DataTransferServiceClient.common_project_path(**expected)
    actual = DataTransferServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DataTransferServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam'}
    path = DataTransferServiceClient.common_location_path(**expected)
    actual = DataTransferServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DataTransferServiceTransport, '_prep_wrapped_messages') as prep:
        client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DataTransferServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DataTransferServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        return 10
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = DataTransferServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = DataTransferServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DataTransferServiceClient, transports.DataTransferServiceGrpcTransport), (DataTransferServiceAsyncClient, transports.DataTransferServiceGrpcAsyncIOTransport)])
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