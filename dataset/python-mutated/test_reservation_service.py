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
from google.oauth2 import service_account
from google.protobuf import any_pb2
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
from google.cloud.bigquery_reservation_v1.services.reservation_service import ReservationServiceAsyncClient, ReservationServiceClient, pagers, transports
from google.cloud.bigquery_reservation_v1.types import reservation as gcbr_reservation
from google.cloud.bigquery_reservation_v1.types import reservation

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert ReservationServiceClient._get_default_mtls_endpoint(None) is None
    assert ReservationServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ReservationServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ReservationServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ReservationServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ReservationServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ReservationServiceClient, 'grpc'), (ReservationServiceAsyncClient, 'grpc_asyncio'), (ReservationServiceClient, 'rest')])
def test_reservation_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('bigqueryreservation.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryreservation.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ReservationServiceGrpcTransport, 'grpc'), (transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ReservationServiceRestTransport, 'rest')])
def test_reservation_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ReservationServiceClient, 'grpc'), (ReservationServiceAsyncClient, 'grpc_asyncio'), (ReservationServiceClient, 'rest')])
def test_reservation_service_client_from_service_account_file(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('bigqueryreservation.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryreservation.googleapis.com')

def test_reservation_service_client_get_transport_class():
    if False:
        return 10
    transport = ReservationServiceClient.get_transport_class()
    available_transports = [transports.ReservationServiceGrpcTransport, transports.ReservationServiceRestTransport]
    assert transport in available_transports
    transport = ReservationServiceClient.get_transport_class('grpc')
    assert transport == transports.ReservationServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ReservationServiceClient, transports.ReservationServiceGrpcTransport, 'grpc'), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ReservationServiceClient, transports.ReservationServiceRestTransport, 'rest')])
@mock.patch.object(ReservationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ReservationServiceClient))
@mock.patch.object(ReservationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ReservationServiceAsyncClient))
def test_reservation_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(ReservationServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ReservationServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ReservationServiceClient, transports.ReservationServiceGrpcTransport, 'grpc', 'true'), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ReservationServiceClient, transports.ReservationServiceGrpcTransport, 'grpc', 'false'), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ReservationServiceClient, transports.ReservationServiceRestTransport, 'rest', 'true'), (ReservationServiceClient, transports.ReservationServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ReservationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ReservationServiceClient))
@mock.patch.object(ReservationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ReservationServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_reservation_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ReservationServiceClient, ReservationServiceAsyncClient])
@mock.patch.object(ReservationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ReservationServiceClient))
@mock.patch.object(ReservationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ReservationServiceAsyncClient))
def test_reservation_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ReservationServiceClient, transports.ReservationServiceGrpcTransport, 'grpc'), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ReservationServiceClient, transports.ReservationServiceRestTransport, 'rest')])
def test_reservation_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ReservationServiceClient, transports.ReservationServiceGrpcTransport, 'grpc', grpc_helpers), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ReservationServiceClient, transports.ReservationServiceRestTransport, 'rest', None)])
def test_reservation_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_reservation_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.bigquery_reservation_v1.services.reservation_service.transports.ReservationServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ReservationServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ReservationServiceClient, transports.ReservationServiceGrpcTransport, 'grpc', grpc_helpers), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_reservation_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('bigqueryreservation.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), scopes=None, default_host='bigqueryreservation.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [gcbr_reservation.CreateReservationRequest, dict])
def test_create_reservation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=gcbr_reservation.Edition.STANDARD)
        response = client.create_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbr_reservation.CreateReservationRequest()
    assert isinstance(response, gcbr_reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == gcbr_reservation.Edition.STANDARD

def test_create_reservation_empty_call():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        client.create_reservation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbr_reservation.CreateReservationRequest()

@pytest.mark.asyncio
async def test_create_reservation_async(transport: str='grpc_asyncio', request_type=gcbr_reservation.CreateReservationRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbr_reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=gcbr_reservation.Edition.STANDARD))
        response = await client.create_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbr_reservation.CreateReservationRequest()
    assert isinstance(response, gcbr_reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == gcbr_reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_create_reservation_async_from_dict():
    await test_create_reservation_async(request_type=dict)

def test_create_reservation_field_headers():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbr_reservation.CreateReservationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation()
        client.create_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_reservation_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbr_reservation.CreateReservationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbr_reservation.Reservation())
        await client.create_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_reservation_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation()
        client.create_reservation(parent='parent_value', reservation=gcbr_reservation.Reservation(name='name_value'), reservation_id='reservation_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].reservation
        mock_val = gcbr_reservation.Reservation(name='name_value')
        assert arg == mock_val
        arg = args[0].reservation_id
        mock_val = 'reservation_id_value'
        assert arg == mock_val

def test_create_reservation_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_reservation(gcbr_reservation.CreateReservationRequest(), parent='parent_value', reservation=gcbr_reservation.Reservation(name='name_value'), reservation_id='reservation_id_value')

@pytest.mark.asyncio
async def test_create_reservation_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbr_reservation.Reservation())
        response = await client.create_reservation(parent='parent_value', reservation=gcbr_reservation.Reservation(name='name_value'), reservation_id='reservation_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].reservation
        mock_val = gcbr_reservation.Reservation(name='name_value')
        assert arg == mock_val
        arg = args[0].reservation_id
        mock_val = 'reservation_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_reservation_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_reservation(gcbr_reservation.CreateReservationRequest(), parent='parent_value', reservation=gcbr_reservation.Reservation(name='name_value'), reservation_id='reservation_id_value')

@pytest.mark.parametrize('request_type', [reservation.ListReservationsRequest, dict])
def test_list_reservations(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.return_value = reservation.ListReservationsResponse(next_page_token='next_page_token_value')
        response = client.list_reservations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListReservationsRequest()
    assert isinstance(response, pagers.ListReservationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_reservations_empty_call():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        client.list_reservations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListReservationsRequest()

@pytest.mark.asyncio
async def test_list_reservations_async(transport: str='grpc_asyncio', request_type=reservation.ListReservationsRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListReservationsResponse(next_page_token='next_page_token_value'))
        response = await client.list_reservations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListReservationsRequest()
    assert isinstance(response, pagers.ListReservationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_reservations_async_from_dict():
    await test_list_reservations_async(request_type=dict)

def test_list_reservations_field_headers():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.ListReservationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.return_value = reservation.ListReservationsResponse()
        client.list_reservations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_reservations_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.ListReservationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListReservationsResponse())
        await client.list_reservations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_reservations_flattened():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.return_value = reservation.ListReservationsResponse()
        client.list_reservations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_reservations_flattened_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_reservations(reservation.ListReservationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_reservations_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.return_value = reservation.ListReservationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListReservationsResponse())
        response = await client.list_reservations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_reservations_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_reservations(reservation.ListReservationsRequest(), parent='parent_value')

def test_list_reservations_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.side_effect = (reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation(), reservation.Reservation()], next_page_token='abc'), reservation.ListReservationsResponse(reservations=[], next_page_token='def'), reservation.ListReservationsResponse(reservations=[reservation.Reservation()], next_page_token='ghi'), reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_reservations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Reservation) for i in results))

def test_list_reservations_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_reservations), '__call__') as call:
        call.side_effect = (reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation(), reservation.Reservation()], next_page_token='abc'), reservation.ListReservationsResponse(reservations=[], next_page_token='def'), reservation.ListReservationsResponse(reservations=[reservation.Reservation()], next_page_token='ghi'), reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation()]), RuntimeError)
        pages = list(client.list_reservations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_reservations_async_pager():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_reservations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation(), reservation.Reservation()], next_page_token='abc'), reservation.ListReservationsResponse(reservations=[], next_page_token='def'), reservation.ListReservationsResponse(reservations=[reservation.Reservation()], next_page_token='ghi'), reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation()]), RuntimeError)
        async_pager = await client.list_reservations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, reservation.Reservation) for i in responses))

@pytest.mark.asyncio
async def test_list_reservations_async_pages():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_reservations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation(), reservation.Reservation()], next_page_token='abc'), reservation.ListReservationsResponse(reservations=[], next_page_token='def'), reservation.ListReservationsResponse(reservations=[reservation.Reservation()], next_page_token='ghi'), reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_reservations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.GetReservationRequest, dict])
def test_get_reservation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        call.return_value = reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response = client.get_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetReservationRequest()
    assert isinstance(response, reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_get_reservation_empty_call():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        client.get_reservation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetReservationRequest()

@pytest.mark.asyncio
async def test_get_reservation_async(transport: str='grpc_asyncio', request_type=reservation.GetReservationRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD))
        response = await client.get_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetReservationRequest()
    assert isinstance(response, reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_get_reservation_async_from_dict():
    await test_get_reservation_async(request_type=dict)

def test_get_reservation_field_headers():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.GetReservationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        call.return_value = reservation.Reservation()
        client.get_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_reservation_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.GetReservationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Reservation())
        await client.get_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_reservation_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        call.return_value = reservation.Reservation()
        client.get_reservation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_reservation_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_reservation(reservation.GetReservationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_reservation_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_reservation), '__call__') as call:
        call.return_value = reservation.Reservation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Reservation())
        response = await client.get_reservation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_reservation_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_reservation(reservation.GetReservationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [reservation.DeleteReservationRequest, dict])
def test_delete_reservation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        call.return_value = None
        response = client.delete_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteReservationRequest()
    assert response is None

def test_delete_reservation_empty_call():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        client.delete_reservation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteReservationRequest()

@pytest.mark.asyncio
async def test_delete_reservation_async(transport: str='grpc_asyncio', request_type=reservation.DeleteReservationRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteReservationRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_reservation_async_from_dict():
    await test_delete_reservation_async(request_type=dict)

def test_delete_reservation_field_headers():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.DeleteReservationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        call.return_value = None
        client.delete_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_reservation_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.DeleteReservationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_reservation_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        call.return_value = None
        client.delete_reservation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_reservation_flattened_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_reservation(reservation.DeleteReservationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_reservation_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_reservation), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_reservation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_reservation_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_reservation(reservation.DeleteReservationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcbr_reservation.UpdateReservationRequest, dict])
def test_update_reservation(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=gcbr_reservation.Edition.STANDARD)
        response = client.update_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbr_reservation.UpdateReservationRequest()
    assert isinstance(response, gcbr_reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == gcbr_reservation.Edition.STANDARD

def test_update_reservation_empty_call():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        client.update_reservation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbr_reservation.UpdateReservationRequest()

@pytest.mark.asyncio
async def test_update_reservation_async(transport: str='grpc_asyncio', request_type=gcbr_reservation.UpdateReservationRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbr_reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=gcbr_reservation.Edition.STANDARD))
        response = await client.update_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbr_reservation.UpdateReservationRequest()
    assert isinstance(response, gcbr_reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == gcbr_reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_update_reservation_async_from_dict():
    await test_update_reservation_async(request_type=dict)

def test_update_reservation_field_headers():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbr_reservation.UpdateReservationRequest()
    request.reservation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation()
        client.update_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'reservation.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_reservation_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbr_reservation.UpdateReservationRequest()
    request.reservation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbr_reservation.Reservation())
        await client.update_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'reservation.name=name_value') in kw['metadata']

def test_update_reservation_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation()
        client.update_reservation(reservation=gcbr_reservation.Reservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].reservation
        mock_val = gcbr_reservation.Reservation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_reservation_flattened_error():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_reservation(gcbr_reservation.UpdateReservationRequest(), reservation=gcbr_reservation.Reservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_reservation_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_reservation), '__call__') as call:
        call.return_value = gcbr_reservation.Reservation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbr_reservation.Reservation())
        response = await client.update_reservation(reservation=gcbr_reservation.Reservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].reservation
        mock_val = gcbr_reservation.Reservation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_reservation_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_reservation(gcbr_reservation.UpdateReservationRequest(), reservation=gcbr_reservation.Reservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [reservation.CreateCapacityCommitmentRequest, dict])
def test_create_capacity_commitment(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response = client.create_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.CreateCapacityCommitmentRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_create_capacity_commitment_empty_call():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        client.create_capacity_commitment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.CreateCapacityCommitmentRequest()

@pytest.mark.asyncio
async def test_create_capacity_commitment_async(transport: str='grpc_asyncio', request_type=reservation.CreateCapacityCommitmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD))
        response = await client.create_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.CreateCapacityCommitmentRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_create_capacity_commitment_async_from_dict():
    await test_create_capacity_commitment_async(request_type=dict)

def test_create_capacity_commitment_field_headers():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.CreateCapacityCommitmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.create_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_capacity_commitment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.CreateCapacityCommitmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        await client.create_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_capacity_commitment_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.create_capacity_commitment(parent='parent_value', capacity_commitment=reservation.CapacityCommitment(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].capacity_commitment
        mock_val = reservation.CapacityCommitment(name='name_value')
        assert arg == mock_val

def test_create_capacity_commitment_flattened_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_capacity_commitment(reservation.CreateCapacityCommitmentRequest(), parent='parent_value', capacity_commitment=reservation.CapacityCommitment(name='name_value'))

@pytest.mark.asyncio
async def test_create_capacity_commitment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        response = await client.create_capacity_commitment(parent='parent_value', capacity_commitment=reservation.CapacityCommitment(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].capacity_commitment
        mock_val = reservation.CapacityCommitment(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_capacity_commitment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_capacity_commitment(reservation.CreateCapacityCommitmentRequest(), parent='parent_value', capacity_commitment=reservation.CapacityCommitment(name='name_value'))

@pytest.mark.parametrize('request_type', [reservation.ListCapacityCommitmentsRequest, dict])
def test_list_capacity_commitments(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.return_value = reservation.ListCapacityCommitmentsResponse(next_page_token='next_page_token_value')
        response = client.list_capacity_commitments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListCapacityCommitmentsRequest()
    assert isinstance(response, pagers.ListCapacityCommitmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_capacity_commitments_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        client.list_capacity_commitments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListCapacityCommitmentsRequest()

@pytest.mark.asyncio
async def test_list_capacity_commitments_async(transport: str='grpc_asyncio', request_type=reservation.ListCapacityCommitmentsRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListCapacityCommitmentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_capacity_commitments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListCapacityCommitmentsRequest()
    assert isinstance(response, pagers.ListCapacityCommitmentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_capacity_commitments_async_from_dict():
    await test_list_capacity_commitments_async(request_type=dict)

def test_list_capacity_commitments_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.ListCapacityCommitmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.return_value = reservation.ListCapacityCommitmentsResponse()
        client.list_capacity_commitments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_capacity_commitments_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.ListCapacityCommitmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListCapacityCommitmentsResponse())
        await client.list_capacity_commitments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_capacity_commitments_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.return_value = reservation.ListCapacityCommitmentsResponse()
        client.list_capacity_commitments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_capacity_commitments_flattened_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_capacity_commitments(reservation.ListCapacityCommitmentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_capacity_commitments_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.return_value = reservation.ListCapacityCommitmentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListCapacityCommitmentsResponse())
        response = await client.list_capacity_commitments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_capacity_commitments_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_capacity_commitments(reservation.ListCapacityCommitmentsRequest(), parent='parent_value')

def test_list_capacity_commitments_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.side_effect = (reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment(), reservation.CapacityCommitment()], next_page_token='abc'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[], next_page_token='def'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment()], next_page_token='ghi'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_capacity_commitments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.CapacityCommitment) for i in results))

def test_list_capacity_commitments_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__') as call:
        call.side_effect = (reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment(), reservation.CapacityCommitment()], next_page_token='abc'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[], next_page_token='def'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment()], next_page_token='ghi'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment()]), RuntimeError)
        pages = list(client.list_capacity_commitments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_capacity_commitments_async_pager():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment(), reservation.CapacityCommitment()], next_page_token='abc'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[], next_page_token='def'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment()], next_page_token='ghi'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment()]), RuntimeError)
        async_pager = await client.list_capacity_commitments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, reservation.CapacityCommitment) for i in responses))

@pytest.mark.asyncio
async def test_list_capacity_commitments_async_pages():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_capacity_commitments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment(), reservation.CapacityCommitment()], next_page_token='abc'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[], next_page_token='def'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment()], next_page_token='ghi'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_capacity_commitments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.GetCapacityCommitmentRequest, dict])
def test_get_capacity_commitment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response = client.get_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetCapacityCommitmentRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_get_capacity_commitment_empty_call():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        client.get_capacity_commitment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetCapacityCommitmentRequest()

@pytest.mark.asyncio
async def test_get_capacity_commitment_async(transport: str='grpc_asyncio', request_type=reservation.GetCapacityCommitmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD))
        response = await client.get_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetCapacityCommitmentRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_get_capacity_commitment_async_from_dict():
    await test_get_capacity_commitment_async(request_type=dict)

def test_get_capacity_commitment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.GetCapacityCommitmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.get_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_capacity_commitment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.GetCapacityCommitmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        await client.get_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_capacity_commitment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.get_capacity_commitment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_capacity_commitment_flattened_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_capacity_commitment(reservation.GetCapacityCommitmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_capacity_commitment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        response = await client.get_capacity_commitment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_capacity_commitment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_capacity_commitment(reservation.GetCapacityCommitmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [reservation.DeleteCapacityCommitmentRequest, dict])
def test_delete_capacity_commitment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        call.return_value = None
        response = client.delete_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteCapacityCommitmentRequest()
    assert response is None

def test_delete_capacity_commitment_empty_call():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        client.delete_capacity_commitment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteCapacityCommitmentRequest()

@pytest.mark.asyncio
async def test_delete_capacity_commitment_async(transport: str='grpc_asyncio', request_type=reservation.DeleteCapacityCommitmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteCapacityCommitmentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_capacity_commitment_async_from_dict():
    await test_delete_capacity_commitment_async(request_type=dict)

def test_delete_capacity_commitment_field_headers():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.DeleteCapacityCommitmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        call.return_value = None
        client.delete_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_capacity_commitment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.DeleteCapacityCommitmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_capacity_commitment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        call.return_value = None
        client.delete_capacity_commitment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_capacity_commitment_flattened_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_capacity_commitment(reservation.DeleteCapacityCommitmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_capacity_commitment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_capacity_commitment), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_capacity_commitment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_capacity_commitment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_capacity_commitment(reservation.DeleteCapacityCommitmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [reservation.UpdateCapacityCommitmentRequest, dict])
def test_update_capacity_commitment(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response = client.update_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateCapacityCommitmentRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_update_capacity_commitment_empty_call():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        client.update_capacity_commitment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateCapacityCommitmentRequest()

@pytest.mark.asyncio
async def test_update_capacity_commitment_async(transport: str='grpc_asyncio', request_type=reservation.UpdateCapacityCommitmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD))
        response = await client.update_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateCapacityCommitmentRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_update_capacity_commitment_async_from_dict():
    await test_update_capacity_commitment_async(request_type=dict)

def test_update_capacity_commitment_field_headers():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.UpdateCapacityCommitmentRequest()
    request.capacity_commitment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.update_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'capacity_commitment.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_capacity_commitment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.UpdateCapacityCommitmentRequest()
    request.capacity_commitment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        await client.update_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'capacity_commitment.name=name_value') in kw['metadata']

def test_update_capacity_commitment_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.update_capacity_commitment(capacity_commitment=reservation.CapacityCommitment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].capacity_commitment
        mock_val = reservation.CapacityCommitment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_capacity_commitment_flattened_error():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_capacity_commitment(reservation.UpdateCapacityCommitmentRequest(), capacity_commitment=reservation.CapacityCommitment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_capacity_commitment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_capacity_commitment), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        response = await client.update_capacity_commitment(capacity_commitment=reservation.CapacityCommitment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].capacity_commitment
        mock_val = reservation.CapacityCommitment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_capacity_commitment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_capacity_commitment(reservation.UpdateCapacityCommitmentRequest(), capacity_commitment=reservation.CapacityCommitment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [reservation.SplitCapacityCommitmentRequest, dict])
def test_split_capacity_commitment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        call.return_value = reservation.SplitCapacityCommitmentResponse()
        response = client.split_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SplitCapacityCommitmentRequest()
    assert isinstance(response, reservation.SplitCapacityCommitmentResponse)

def test_split_capacity_commitment_empty_call():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        client.split_capacity_commitment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SplitCapacityCommitmentRequest()

@pytest.mark.asyncio
async def test_split_capacity_commitment_async(transport: str='grpc_asyncio', request_type=reservation.SplitCapacityCommitmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SplitCapacityCommitmentResponse())
        response = await client.split_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SplitCapacityCommitmentRequest()
    assert isinstance(response, reservation.SplitCapacityCommitmentResponse)

@pytest.mark.asyncio
async def test_split_capacity_commitment_async_from_dict():
    await test_split_capacity_commitment_async(request_type=dict)

def test_split_capacity_commitment_field_headers():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.SplitCapacityCommitmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        call.return_value = reservation.SplitCapacityCommitmentResponse()
        client.split_capacity_commitment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_split_capacity_commitment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.SplitCapacityCommitmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SplitCapacityCommitmentResponse())
        await client.split_capacity_commitment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_split_capacity_commitment_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        call.return_value = reservation.SplitCapacityCommitmentResponse()
        client.split_capacity_commitment(name='name_value', slot_count=1098)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].slot_count
        mock_val = 1098
        assert arg == mock_val

def test_split_capacity_commitment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.split_capacity_commitment(reservation.SplitCapacityCommitmentRequest(), name='name_value', slot_count=1098)

@pytest.mark.asyncio
async def test_split_capacity_commitment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.split_capacity_commitment), '__call__') as call:
        call.return_value = reservation.SplitCapacityCommitmentResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SplitCapacityCommitmentResponse())
        response = await client.split_capacity_commitment(name='name_value', slot_count=1098)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].slot_count
        mock_val = 1098
        assert arg == mock_val

@pytest.mark.asyncio
async def test_split_capacity_commitment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.split_capacity_commitment(reservation.SplitCapacityCommitmentRequest(), name='name_value', slot_count=1098)

@pytest.mark.parametrize('request_type', [reservation.MergeCapacityCommitmentsRequest, dict])
def test_merge_capacity_commitments(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        call.return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response = client.merge_capacity_commitments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.MergeCapacityCommitmentsRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_merge_capacity_commitments_empty_call():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        client.merge_capacity_commitments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.MergeCapacityCommitmentsRequest()

@pytest.mark.asyncio
async def test_merge_capacity_commitments_async(transport: str='grpc_asyncio', request_type=reservation.MergeCapacityCommitmentsRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD))
        response = await client.merge_capacity_commitments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.MergeCapacityCommitmentsRequest()
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.asyncio
async def test_merge_capacity_commitments_async_from_dict():
    await test_merge_capacity_commitments_async(request_type=dict)

def test_merge_capacity_commitments_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.MergeCapacityCommitmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.merge_capacity_commitments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_merge_capacity_commitments_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.MergeCapacityCommitmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        await client.merge_capacity_commitments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_merge_capacity_commitments_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        client.merge_capacity_commitments(parent='parent_value', capacity_commitment_ids=['capacity_commitment_ids_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].capacity_commitment_ids
        mock_val = ['capacity_commitment_ids_value']
        assert arg == mock_val

def test_merge_capacity_commitments_flattened_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.merge_capacity_commitments(reservation.MergeCapacityCommitmentsRequest(), parent='parent_value', capacity_commitment_ids=['capacity_commitment_ids_value'])

@pytest.mark.asyncio
async def test_merge_capacity_commitments_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.merge_capacity_commitments), '__call__') as call:
        call.return_value = reservation.CapacityCommitment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.CapacityCommitment())
        response = await client.merge_capacity_commitments(parent='parent_value', capacity_commitment_ids=['capacity_commitment_ids_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].capacity_commitment_ids
        mock_val = ['capacity_commitment_ids_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_merge_capacity_commitments_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.merge_capacity_commitments(reservation.MergeCapacityCommitmentsRequest(), parent='parent_value', capacity_commitment_ids=['capacity_commitment_ids_value'])

@pytest.mark.parametrize('request_type', [reservation.CreateAssignmentRequest, dict])
def test_create_assignment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        call.return_value = reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING)
        response = client.create_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.CreateAssignmentRequest()
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

def test_create_assignment_empty_call():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        client.create_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.CreateAssignmentRequest()

@pytest.mark.asyncio
async def test_create_assignment_async(transport: str='grpc_asyncio', request_type=reservation.CreateAssignmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING))
        response = await client.create_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.CreateAssignmentRequest()
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

@pytest.mark.asyncio
async def test_create_assignment_async_from_dict():
    await test_create_assignment_async(request_type=dict)

def test_create_assignment_field_headers():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.CreateAssignmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        client.create_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_assignment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.CreateAssignmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment())
        await client.create_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_assignment_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        client.create_assignment(parent='parent_value', assignment=reservation.Assignment(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].assignment
        mock_val = reservation.Assignment(name='name_value')
        assert arg == mock_val

def test_create_assignment_flattened_error():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_assignment(reservation.CreateAssignmentRequest(), parent='parent_value', assignment=reservation.Assignment(name='name_value'))

@pytest.mark.asyncio
async def test_create_assignment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment())
        response = await client.create_assignment(parent='parent_value', assignment=reservation.Assignment(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].assignment
        mock_val = reservation.Assignment(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_assignment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_assignment(reservation.CreateAssignmentRequest(), parent='parent_value', assignment=reservation.Assignment(name='name_value'))

@pytest.mark.parametrize('request_type', [reservation.ListAssignmentsRequest, dict])
def test_list_assignments(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.return_value = reservation.ListAssignmentsResponse(next_page_token='next_page_token_value')
        response = client.list_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListAssignmentsRequest()
    assert isinstance(response, pagers.ListAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_assignments_empty_call():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        client.list_assignments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListAssignmentsRequest()

@pytest.mark.asyncio
async def test_list_assignments_async(transport: str='grpc_asyncio', request_type=reservation.ListAssignmentsRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListAssignmentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.ListAssignmentsRequest()
    assert isinstance(response, pagers.ListAssignmentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_assignments_async_from_dict():
    await test_list_assignments_async(request_type=dict)

def test_list_assignments_field_headers():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.ListAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.return_value = reservation.ListAssignmentsResponse()
        client.list_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_assignments_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.ListAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListAssignmentsResponse())
        await client.list_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_assignments_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.return_value = reservation.ListAssignmentsResponse()
        client.list_assignments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_assignments_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_assignments(reservation.ListAssignmentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_assignments_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.return_value = reservation.ListAssignmentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.ListAssignmentsResponse())
        response = await client.list_assignments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_assignments_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_assignments(reservation.ListAssignmentsRequest(), parent='parent_value')

def test_list_assignments_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.side_effect = (reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.ListAssignmentsResponse(assignments=[], next_page_token='def'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_assignments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Assignment) for i in results))

def test_list_assignments_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_assignments), '__call__') as call:
        call.side_effect = (reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.ListAssignmentsResponse(assignments=[], next_page_token='def'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        pages = list(client.list_assignments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_assignments_async_pager():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.ListAssignmentsResponse(assignments=[], next_page_token='def'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        async_pager = await client.list_assignments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, reservation.Assignment) for i in responses))

@pytest.mark.asyncio
async def test_list_assignments_async_pages():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.ListAssignmentsResponse(assignments=[], next_page_token='def'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_assignments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.DeleteAssignmentRequest, dict])
def test_delete_assignment(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        call.return_value = None
        response = client.delete_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteAssignmentRequest()
    assert response is None

def test_delete_assignment_empty_call():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        client.delete_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteAssignmentRequest()

@pytest.mark.asyncio
async def test_delete_assignment_async(transport: str='grpc_asyncio', request_type=reservation.DeleteAssignmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.DeleteAssignmentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_assignment_async_from_dict():
    await test_delete_assignment_async(request_type=dict)

def test_delete_assignment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.DeleteAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        call.return_value = None
        client.delete_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_assignment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.DeleteAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_assignment_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        call.return_value = None
        client.delete_assignment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_assignment_flattened_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_assignment(reservation.DeleteAssignmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_assignment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_assignment), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_assignment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_assignment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_assignment(reservation.DeleteAssignmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [reservation.SearchAssignmentsRequest, dict])
def test_search_assignments(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.return_value = reservation.SearchAssignmentsResponse(next_page_token='next_page_token_value')
        response = client.search_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SearchAssignmentsRequest()
    assert isinstance(response, pagers.SearchAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_assignments_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        client.search_assignments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SearchAssignmentsRequest()

@pytest.mark.asyncio
async def test_search_assignments_async(transport: str='grpc_asyncio', request_type=reservation.SearchAssignmentsRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SearchAssignmentsResponse(next_page_token='next_page_token_value'))
        response = await client.search_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SearchAssignmentsRequest()
    assert isinstance(response, pagers.SearchAssignmentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_assignments_async_from_dict():
    await test_search_assignments_async(request_type=dict)

def test_search_assignments_field_headers():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.SearchAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.return_value = reservation.SearchAssignmentsResponse()
        client.search_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_assignments_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.SearchAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SearchAssignmentsResponse())
        await client.search_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_assignments_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.return_value = reservation.SearchAssignmentsResponse()
        client.search_assignments(parent='parent_value', query='query_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

def test_search_assignments_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_assignments(reservation.SearchAssignmentsRequest(), parent='parent_value', query='query_value')

@pytest.mark.asyncio
async def test_search_assignments_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.return_value = reservation.SearchAssignmentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SearchAssignmentsResponse())
        response = await client.search_assignments(parent='parent_value', query='query_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_assignments_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_assignments(reservation.SearchAssignmentsRequest(), parent='parent_value', query='query_value')

def test_search_assignments_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.side_effect = (reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.search_assignments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Assignment) for i in results))

def test_search_assignments_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_assignments), '__call__') as call:
        call.side_effect = (reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        pages = list(client.search_assignments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_assignments_async_pager():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        async_pager = await client.search_assignments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, reservation.Assignment) for i in responses))

@pytest.mark.asyncio
async def test_search_assignments_async_pages():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_assignments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.SearchAllAssignmentsRequest, dict])
def test_search_all_assignments(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.return_value = reservation.SearchAllAssignmentsResponse(next_page_token='next_page_token_value')
        response = client.search_all_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SearchAllAssignmentsRequest()
    assert isinstance(response, pagers.SearchAllAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_all_assignments_empty_call():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        client.search_all_assignments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SearchAllAssignmentsRequest()

@pytest.mark.asyncio
async def test_search_all_assignments_async(transport: str='grpc_asyncio', request_type=reservation.SearchAllAssignmentsRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SearchAllAssignmentsResponse(next_page_token='next_page_token_value'))
        response = await client.search_all_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.SearchAllAssignmentsRequest()
    assert isinstance(response, pagers.SearchAllAssignmentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_all_assignments_async_from_dict():
    await test_search_all_assignments_async(request_type=dict)

def test_search_all_assignments_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.SearchAllAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.return_value = reservation.SearchAllAssignmentsResponse()
        client.search_all_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_all_assignments_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.SearchAllAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SearchAllAssignmentsResponse())
        await client.search_all_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_all_assignments_flattened():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.return_value = reservation.SearchAllAssignmentsResponse()
        client.search_all_assignments(parent='parent_value', query='query_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

def test_search_all_assignments_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_all_assignments(reservation.SearchAllAssignmentsRequest(), parent='parent_value', query='query_value')

@pytest.mark.asyncio
async def test_search_all_assignments_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.return_value = reservation.SearchAllAssignmentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.SearchAllAssignmentsResponse())
        response = await client.search_all_assignments(parent='parent_value', query='query_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_all_assignments_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_all_assignments(reservation.SearchAllAssignmentsRequest(), parent='parent_value', query='query_value')

def test_search_all_assignments_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.side_effect = (reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAllAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.search_all_assignments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Assignment) for i in results))

def test_search_all_assignments_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__') as call:
        call.side_effect = (reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAllAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        pages = list(client.search_all_assignments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_all_assignments_async_pager():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAllAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        async_pager = await client.search_all_assignments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, reservation.Assignment) for i in responses))

@pytest.mark.asyncio
async def test_search_all_assignments_async_pages():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_all_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAllAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_all_assignments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.MoveAssignmentRequest, dict])
def test_move_assignment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        call.return_value = reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING)
        response = client.move_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.MoveAssignmentRequest()
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

def test_move_assignment_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        client.move_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.MoveAssignmentRequest()

@pytest.mark.asyncio
async def test_move_assignment_async(transport: str='grpc_asyncio', request_type=reservation.MoveAssignmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING))
        response = await client.move_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.MoveAssignmentRequest()
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

@pytest.mark.asyncio
async def test_move_assignment_async_from_dict():
    await test_move_assignment_async(request_type=dict)

def test_move_assignment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.MoveAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        client.move_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_move_assignment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.MoveAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment())
        await client.move_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_move_assignment_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        client.move_assignment(name='name_value', destination_id='destination_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].destination_id
        mock_val = 'destination_id_value'
        assert arg == mock_val

def test_move_assignment_flattened_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.move_assignment(reservation.MoveAssignmentRequest(), name='name_value', destination_id='destination_id_value')

@pytest.mark.asyncio
async def test_move_assignment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.move_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment())
        response = await client.move_assignment(name='name_value', destination_id='destination_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].destination_id
        mock_val = 'destination_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_move_assignment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.move_assignment(reservation.MoveAssignmentRequest(), name='name_value', destination_id='destination_id_value')

@pytest.mark.parametrize('request_type', [reservation.UpdateAssignmentRequest, dict])
def test_update_assignment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        call.return_value = reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING)
        response = client.update_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateAssignmentRequest()
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

def test_update_assignment_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        client.update_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateAssignmentRequest()

@pytest.mark.asyncio
async def test_update_assignment_async(transport: str='grpc_asyncio', request_type=reservation.UpdateAssignmentRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING))
        response = await client.update_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateAssignmentRequest()
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

@pytest.mark.asyncio
async def test_update_assignment_async_from_dict():
    await test_update_assignment_async(request_type=dict)

def test_update_assignment_field_headers():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.UpdateAssignmentRequest()
    request.assignment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        client.update_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'assignment.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_assignment_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.UpdateAssignmentRequest()
    request.assignment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment())
        await client.update_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'assignment.name=name_value') in kw['metadata']

def test_update_assignment_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        client.update_assignment(assignment=reservation.Assignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].assignment
        mock_val = reservation.Assignment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_assignment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_assignment(reservation.UpdateAssignmentRequest(), assignment=reservation.Assignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_assignment_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_assignment), '__call__') as call:
        call.return_value = reservation.Assignment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.Assignment())
        response = await client.update_assignment(assignment=reservation.Assignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].assignment
        mock_val = reservation.Assignment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_assignment_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_assignment(reservation.UpdateAssignmentRequest(), assignment=reservation.Assignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [reservation.GetBiReservationRequest, dict])
def test_get_bi_reservation(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation(name='name_value', size=443)
        response = client.get_bi_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetBiReservationRequest()
    assert isinstance(response, reservation.BiReservation)
    assert response.name == 'name_value'
    assert response.size == 443

def test_get_bi_reservation_empty_call():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        client.get_bi_reservation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetBiReservationRequest()

@pytest.mark.asyncio
async def test_get_bi_reservation_async(transport: str='grpc_asyncio', request_type=reservation.GetBiReservationRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.BiReservation(name='name_value', size=443))
        response = await client.get_bi_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.GetBiReservationRequest()
    assert isinstance(response, reservation.BiReservation)
    assert response.name == 'name_value'
    assert response.size == 443

@pytest.mark.asyncio
async def test_get_bi_reservation_async_from_dict():
    await test_get_bi_reservation_async(request_type=dict)

def test_get_bi_reservation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.GetBiReservationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation()
        client.get_bi_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_bi_reservation_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.GetBiReservationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.BiReservation())
        await client.get_bi_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_bi_reservation_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation()
        client.get_bi_reservation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_bi_reservation_flattened_error():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_bi_reservation(reservation.GetBiReservationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_bi_reservation_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.BiReservation())
        response = await client.get_bi_reservation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_bi_reservation_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_bi_reservation(reservation.GetBiReservationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [reservation.UpdateBiReservationRequest, dict])
def test_update_bi_reservation(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation(name='name_value', size=443)
        response = client.update_bi_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateBiReservationRequest()
    assert isinstance(response, reservation.BiReservation)
    assert response.name == 'name_value'
    assert response.size == 443

def test_update_bi_reservation_empty_call():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        client.update_bi_reservation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateBiReservationRequest()

@pytest.mark.asyncio
async def test_update_bi_reservation_async(transport: str='grpc_asyncio', request_type=reservation.UpdateBiReservationRequest):
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.BiReservation(name='name_value', size=443))
        response = await client.update_bi_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == reservation.UpdateBiReservationRequest()
    assert isinstance(response, reservation.BiReservation)
    assert response.name == 'name_value'
    assert response.size == 443

@pytest.mark.asyncio
async def test_update_bi_reservation_async_from_dict():
    await test_update_bi_reservation_async(request_type=dict)

def test_update_bi_reservation_field_headers():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.UpdateBiReservationRequest()
    request.bi_reservation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation()
        client.update_bi_reservation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'bi_reservation.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_bi_reservation_field_headers_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = reservation.UpdateBiReservationRequest()
    request.bi_reservation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.BiReservation())
        await client.update_bi_reservation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'bi_reservation.name=name_value') in kw['metadata']

def test_update_bi_reservation_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation()
        client.update_bi_reservation(bi_reservation=reservation.BiReservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].bi_reservation
        mock_val = reservation.BiReservation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_bi_reservation_flattened_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_bi_reservation(reservation.UpdateBiReservationRequest(), bi_reservation=reservation.BiReservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_bi_reservation_flattened_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_bi_reservation), '__call__') as call:
        call.return_value = reservation.BiReservation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(reservation.BiReservation())
        response = await client.update_bi_reservation(bi_reservation=reservation.BiReservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].bi_reservation
        mock_val = reservation.BiReservation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_bi_reservation_flattened_error_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_bi_reservation(reservation.UpdateBiReservationRequest(), bi_reservation=reservation.BiReservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [gcbr_reservation.CreateReservationRequest, dict])
def test_create_reservation_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['reservation'] = {'name': 'name_value', 'slot_capacity': 1391, 'ignore_idle_slots': True, 'autoscale': {'current_slots': 1431, 'max_slots': 986}, 'concurrency': 1195, 'creation_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'multi_region_auxiliary': True, 'edition': 1}
    test_field = gcbr_reservation.CreateReservationRequest.meta.fields['reservation']

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
    for (field, value) in request_init['reservation'].items():
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
                for i in range(0, len(request_init['reservation'][field])):
                    del request_init['reservation'][field][i][subfield]
            else:
                del request_init['reservation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcbr_reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=gcbr_reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbr_reservation.Reservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_reservation(request)
    assert isinstance(response, gcbr_reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == gcbr_reservation.Edition.STANDARD

def test_create_reservation_rest_required_fields(request_type=gcbr_reservation.CreateReservationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_reservation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('reservation_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcbr_reservation.Reservation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcbr_reservation.Reservation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_reservation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_reservation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_reservation._get_unset_required_fields({})
    assert set(unset_fields) == set(('reservationId',)) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_reservation_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_create_reservation') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_create_reservation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcbr_reservation.CreateReservationRequest.pb(gcbr_reservation.CreateReservationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcbr_reservation.Reservation.to_json(gcbr_reservation.Reservation())
        request = gcbr_reservation.CreateReservationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcbr_reservation.Reservation()
        client.create_reservation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_reservation_rest_bad_request(transport: str='rest', request_type=gcbr_reservation.CreateReservationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_reservation(request)

def test_create_reservation_rest_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcbr_reservation.Reservation()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', reservation=gcbr_reservation.Reservation(name='name_value'), reservation_id='reservation_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbr_reservation.Reservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_reservation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/reservations' % client.transport._host, args[1])

def test_create_reservation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_reservation(gcbr_reservation.CreateReservationRequest(), parent='parent_value', reservation=gcbr_reservation.Reservation(name='name_value'), reservation_id='reservation_id_value')

def test_create_reservation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.ListReservationsRequest, dict])
def test_list_reservations_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.ListReservationsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.ListReservationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_reservations(request)
    assert isinstance(response, pagers.ListReservationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_reservations_rest_required_fields(request_type=reservation.ListReservationsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_reservations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_reservations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.ListReservationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.ListReservationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_reservations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_reservations_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_reservations._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_reservations_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_list_reservations') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_list_reservations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.ListReservationsRequest.pb(reservation.ListReservationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.ListReservationsResponse.to_json(reservation.ListReservationsResponse())
        request = reservation.ListReservationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.ListReservationsResponse()
        client.list_reservations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_reservations_rest_bad_request(transport: str='rest', request_type=reservation.ListReservationsRequest):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_reservations(request)

def test_list_reservations_rest_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.ListReservationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.ListReservationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_reservations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/reservations' % client.transport._host, args[1])

def test_list_reservations_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_reservations(reservation.ListReservationsRequest(), parent='parent_value')

def test_list_reservations_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation(), reservation.Reservation()], next_page_token='abc'), reservation.ListReservationsResponse(reservations=[], next_page_token='def'), reservation.ListReservationsResponse(reservations=[reservation.Reservation()], next_page_token='ghi'), reservation.ListReservationsResponse(reservations=[reservation.Reservation(), reservation.Reservation()]))
        response = response + response
        response = tuple((reservation.ListReservationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_reservations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Reservation) for i in results))
        pages = list(client.list_reservations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.GetReservationRequest, dict])
def test_get_reservation_rest(request_type):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Reservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_reservation(request)
    assert isinstance(response, reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_get_reservation_rest_required_fields(request_type=reservation.GetReservationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.Reservation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.Reservation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_reservation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_reservation_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_reservation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_reservation_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_get_reservation') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_get_reservation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.GetReservationRequest.pb(reservation.GetReservationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.Reservation.to_json(reservation.Reservation())
        request = reservation.GetReservationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.Reservation()
        client.get_reservation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_reservation_rest_bad_request(transport: str='rest', request_type=reservation.GetReservationRequest):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_reservation(request)

def test_get_reservation_rest_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Reservation()
        sample_request = {'name': 'projects/sample1/locations/sample2/reservations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Reservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_reservation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/reservations/*}' % client.transport._host, args[1])

def test_get_reservation_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_reservation(reservation.GetReservationRequest(), name='name_value')

def test_get_reservation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.DeleteReservationRequest, dict])
def test_delete_reservation_rest(request_type):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_reservation(request)
    assert response is None

def test_delete_reservation_rest_required_fields(request_type=reservation.DeleteReservationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_reservation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_reservation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_reservation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_reservation_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_delete_reservation') as pre:
        pre.assert_not_called()
        pb_message = reservation.DeleteReservationRequest.pb(reservation.DeleteReservationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = reservation.DeleteReservationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_reservation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_reservation_rest_bad_request(transport: str='rest', request_type=reservation.DeleteReservationRequest):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_reservation(request)

def test_delete_reservation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/reservations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_reservation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/reservations/*}' % client.transport._host, args[1])

def test_delete_reservation_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_reservation(reservation.DeleteReservationRequest(), name='name_value')

def test_delete_reservation_rest_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcbr_reservation.UpdateReservationRequest, dict])
def test_update_reservation_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'reservation': {'name': 'projects/sample1/locations/sample2/reservations/sample3'}}
    request_init['reservation'] = {'name': 'projects/sample1/locations/sample2/reservations/sample3', 'slot_capacity': 1391, 'ignore_idle_slots': True, 'autoscale': {'current_slots': 1431, 'max_slots': 986}, 'concurrency': 1195, 'creation_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'multi_region_auxiliary': True, 'edition': 1}
    test_field = gcbr_reservation.UpdateReservationRequest.meta.fields['reservation']

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
    for (field, value) in request_init['reservation'].items():
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
                for i in range(0, len(request_init['reservation'][field])):
                    del request_init['reservation'][field][i][subfield]
            else:
                del request_init['reservation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcbr_reservation.Reservation(name='name_value', slot_capacity=1391, ignore_idle_slots=True, concurrency=1195, multi_region_auxiliary=True, edition=gcbr_reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbr_reservation.Reservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_reservation(request)
    assert isinstance(response, gcbr_reservation.Reservation)
    assert response.name == 'name_value'
    assert response.slot_capacity == 1391
    assert response.ignore_idle_slots is True
    assert response.concurrency == 1195
    assert response.multi_region_auxiliary is True
    assert response.edition == gcbr_reservation.Edition.STANDARD

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_reservation_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_update_reservation') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_update_reservation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcbr_reservation.UpdateReservationRequest.pb(gcbr_reservation.UpdateReservationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcbr_reservation.Reservation.to_json(gcbr_reservation.Reservation())
        request = gcbr_reservation.UpdateReservationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcbr_reservation.Reservation()
        client.update_reservation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_reservation_rest_bad_request(transport: str='rest', request_type=gcbr_reservation.UpdateReservationRequest):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'reservation': {'name': 'projects/sample1/locations/sample2/reservations/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_reservation(request)

def test_update_reservation_rest_flattened():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcbr_reservation.Reservation()
        sample_request = {'reservation': {'name': 'projects/sample1/locations/sample2/reservations/sample3'}}
        mock_args = dict(reservation=gcbr_reservation.Reservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbr_reservation.Reservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_reservation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{reservation.name=projects/*/locations/*/reservations/*}' % client.transport._host, args[1])

def test_update_reservation_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_reservation(gcbr_reservation.UpdateReservationRequest(), reservation=gcbr_reservation.Reservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_reservation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.CreateCapacityCommitmentRequest, dict])
def test_create_capacity_commitment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['capacity_commitment'] = {'name': 'name_value', 'slot_count': 1098, 'plan': 3, 'state': 1, 'commitment_start_time': {'seconds': 751, 'nanos': 543}, 'commitment_end_time': {}, 'failure_status': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'renewal_plan': 3, 'multi_region_auxiliary': True, 'edition': 1}
    test_field = reservation.CreateCapacityCommitmentRequest.meta.fields['capacity_commitment']

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
    for (field, value) in request_init['capacity_commitment'].items():
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
                for i in range(0, len(request_init['capacity_commitment'][field])):
                    del request_init['capacity_commitment'][field][i][subfield]
            else:
                del request_init['capacity_commitment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_capacity_commitment(request)
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_create_capacity_commitment_rest_required_fields(request_type=reservation.CreateCapacityCommitmentRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_capacity_commitment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_capacity_commitment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('capacity_commitment_id', 'enforce_single_admin_project_per_org'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.CapacityCommitment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.CapacityCommitment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_capacity_commitment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_capacity_commitment_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_capacity_commitment._get_unset_required_fields({})
    assert set(unset_fields) == set(('capacityCommitmentId', 'enforceSingleAdminProjectPerOrg')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_capacity_commitment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_create_capacity_commitment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_create_capacity_commitment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.CreateCapacityCommitmentRequest.pb(reservation.CreateCapacityCommitmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.CapacityCommitment.to_json(reservation.CapacityCommitment())
        request = reservation.CreateCapacityCommitmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.CapacityCommitment()
        client.create_capacity_commitment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_capacity_commitment_rest_bad_request(transport: str='rest', request_type=reservation.CreateCapacityCommitmentRequest):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_capacity_commitment(request)

def test_create_capacity_commitment_rest_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', capacity_commitment=reservation.CapacityCommitment(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_capacity_commitment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/capacityCommitments' % client.transport._host, args[1])

def test_create_capacity_commitment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_capacity_commitment(reservation.CreateCapacityCommitmentRequest(), parent='parent_value', capacity_commitment=reservation.CapacityCommitment(name='name_value'))

def test_create_capacity_commitment_rest_error():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.ListCapacityCommitmentsRequest, dict])
def test_list_capacity_commitments_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.ListCapacityCommitmentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.ListCapacityCommitmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_capacity_commitments(request)
    assert isinstance(response, pagers.ListCapacityCommitmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_capacity_commitments_rest_required_fields(request_type=reservation.ListCapacityCommitmentsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_capacity_commitments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_capacity_commitments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.ListCapacityCommitmentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.ListCapacityCommitmentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_capacity_commitments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_capacity_commitments_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_capacity_commitments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_capacity_commitments_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_list_capacity_commitments') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_list_capacity_commitments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.ListCapacityCommitmentsRequest.pb(reservation.ListCapacityCommitmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.ListCapacityCommitmentsResponse.to_json(reservation.ListCapacityCommitmentsResponse())
        request = reservation.ListCapacityCommitmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.ListCapacityCommitmentsResponse()
        client.list_capacity_commitments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_capacity_commitments_rest_bad_request(transport: str='rest', request_type=reservation.ListCapacityCommitmentsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_capacity_commitments(request)

def test_list_capacity_commitments_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.ListCapacityCommitmentsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.ListCapacityCommitmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_capacity_commitments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/capacityCommitments' % client.transport._host, args[1])

def test_list_capacity_commitments_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_capacity_commitments(reservation.ListCapacityCommitmentsRequest(), parent='parent_value')

def test_list_capacity_commitments_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment(), reservation.CapacityCommitment()], next_page_token='abc'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[], next_page_token='def'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment()], next_page_token='ghi'), reservation.ListCapacityCommitmentsResponse(capacity_commitments=[reservation.CapacityCommitment(), reservation.CapacityCommitment()]))
        response = response + response
        response = tuple((reservation.ListCapacityCommitmentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_capacity_commitments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.CapacityCommitment) for i in results))
        pages = list(client.list_capacity_commitments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.GetCapacityCommitmentRequest, dict])
def test_get_capacity_commitment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_capacity_commitment(request)
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

def test_get_capacity_commitment_rest_required_fields(request_type=reservation.GetCapacityCommitmentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_capacity_commitment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_capacity_commitment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.CapacityCommitment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.CapacityCommitment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_capacity_commitment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_capacity_commitment_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_capacity_commitment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_capacity_commitment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_get_capacity_commitment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_get_capacity_commitment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.GetCapacityCommitmentRequest.pb(reservation.GetCapacityCommitmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.CapacityCommitment.to_json(reservation.CapacityCommitment())
        request = reservation.GetCapacityCommitmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.CapacityCommitment()
        client.get_capacity_commitment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_capacity_commitment_rest_bad_request(transport: str='rest', request_type=reservation.GetCapacityCommitmentRequest):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_capacity_commitment(request)

def test_get_capacity_commitment_rest_flattened():
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment()
        sample_request = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_capacity_commitment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/capacityCommitments/*}' % client.transport._host, args[1])

def test_get_capacity_commitment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_capacity_commitment(reservation.GetCapacityCommitmentRequest(), name='name_value')

def test_get_capacity_commitment_rest_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.DeleteCapacityCommitmentRequest, dict])
def test_delete_capacity_commitment_rest(request_type):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_capacity_commitment(request)
    assert response is None

def test_delete_capacity_commitment_rest_required_fields(request_type=reservation.DeleteCapacityCommitmentRequest):
    if False:
        return 10
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_capacity_commitment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_capacity_commitment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_capacity_commitment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_capacity_commitment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_capacity_commitment._get_unset_required_fields({})
    assert set(unset_fields) == set(('force',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_capacity_commitment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_delete_capacity_commitment') as pre:
        pre.assert_not_called()
        pb_message = reservation.DeleteCapacityCommitmentRequest.pb(reservation.DeleteCapacityCommitmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = reservation.DeleteCapacityCommitmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_capacity_commitment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_capacity_commitment_rest_bad_request(transport: str='rest', request_type=reservation.DeleteCapacityCommitmentRequest):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_capacity_commitment(request)

def test_delete_capacity_commitment_rest_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_capacity_commitment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/capacityCommitments/*}' % client.transport._host, args[1])

def test_delete_capacity_commitment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_capacity_commitment(reservation.DeleteCapacityCommitmentRequest(), name='name_value')

def test_delete_capacity_commitment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.UpdateCapacityCommitmentRequest, dict])
def test_update_capacity_commitment_rest(request_type):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'capacity_commitment': {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}}
    request_init['capacity_commitment'] = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3', 'slot_count': 1098, 'plan': 3, 'state': 1, 'commitment_start_time': {'seconds': 751, 'nanos': 543}, 'commitment_end_time': {}, 'failure_status': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'renewal_plan': 3, 'multi_region_auxiliary': True, 'edition': 1}
    test_field = reservation.UpdateCapacityCommitmentRequest.meta.fields['capacity_commitment']

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
    for (field, value) in request_init['capacity_commitment'].items():
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
                for i in range(0, len(request_init['capacity_commitment'][field])):
                    del request_init['capacity_commitment'][field][i][subfield]
            else:
                del request_init['capacity_commitment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_capacity_commitment(request)
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_capacity_commitment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_update_capacity_commitment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_update_capacity_commitment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.UpdateCapacityCommitmentRequest.pb(reservation.UpdateCapacityCommitmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.CapacityCommitment.to_json(reservation.CapacityCommitment())
        request = reservation.UpdateCapacityCommitmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.CapacityCommitment()
        client.update_capacity_commitment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_capacity_commitment_rest_bad_request(transport: str='rest', request_type=reservation.UpdateCapacityCommitmentRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'capacity_commitment': {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_capacity_commitment(request)

def test_update_capacity_commitment_rest_flattened():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment()
        sample_request = {'capacity_commitment': {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}}
        mock_args = dict(capacity_commitment=reservation.CapacityCommitment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_capacity_commitment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{capacity_commitment.name=projects/*/locations/*/capacityCommitments/*}' % client.transport._host, args[1])

def test_update_capacity_commitment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_capacity_commitment(reservation.UpdateCapacityCommitmentRequest(), capacity_commitment=reservation.CapacityCommitment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_capacity_commitment_rest_error():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.SplitCapacityCommitmentRequest, dict])
def test_split_capacity_commitment_rest(request_type):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.SplitCapacityCommitmentResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.SplitCapacityCommitmentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.split_capacity_commitment(request)
    assert isinstance(response, reservation.SplitCapacityCommitmentResponse)

def test_split_capacity_commitment_rest_required_fields(request_type=reservation.SplitCapacityCommitmentRequest):
    if False:
        return 10
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).split_capacity_commitment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).split_capacity_commitment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.SplitCapacityCommitmentResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.SplitCapacityCommitmentResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.split_capacity_commitment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_split_capacity_commitment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.split_capacity_commitment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_split_capacity_commitment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_split_capacity_commitment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_split_capacity_commitment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.SplitCapacityCommitmentRequest.pb(reservation.SplitCapacityCommitmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.SplitCapacityCommitmentResponse.to_json(reservation.SplitCapacityCommitmentResponse())
        request = reservation.SplitCapacityCommitmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.SplitCapacityCommitmentResponse()
        client.split_capacity_commitment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_split_capacity_commitment_rest_bad_request(transport: str='rest', request_type=reservation.SplitCapacityCommitmentRequest):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.split_capacity_commitment(request)

def test_split_capacity_commitment_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.SplitCapacityCommitmentResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/capacityCommitments/sample3'}
        mock_args = dict(name='name_value', slot_count=1098)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.SplitCapacityCommitmentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.split_capacity_commitment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/capacityCommitments/*}:split' % client.transport._host, args[1])

def test_split_capacity_commitment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.split_capacity_commitment(reservation.SplitCapacityCommitmentRequest(), name='name_value', slot_count=1098)

def test_split_capacity_commitment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.MergeCapacityCommitmentsRequest, dict])
def test_merge_capacity_commitments_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment(name='name_value', slot_count=1098, plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, state=reservation.CapacityCommitment.State.PENDING, renewal_plan=reservation.CapacityCommitment.CommitmentPlan.FLEX, multi_region_auxiliary=True, edition=reservation.Edition.STANDARD)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.merge_capacity_commitments(request)
    assert isinstance(response, reservation.CapacityCommitment)
    assert response.name == 'name_value'
    assert response.slot_count == 1098
    assert response.plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.state == reservation.CapacityCommitment.State.PENDING
    assert response.renewal_plan == reservation.CapacityCommitment.CommitmentPlan.FLEX
    assert response.multi_region_auxiliary is True
    assert response.edition == reservation.Edition.STANDARD

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_merge_capacity_commitments_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_merge_capacity_commitments') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_merge_capacity_commitments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.MergeCapacityCommitmentsRequest.pb(reservation.MergeCapacityCommitmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.CapacityCommitment.to_json(reservation.CapacityCommitment())
        request = reservation.MergeCapacityCommitmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.CapacityCommitment()
        client.merge_capacity_commitments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_merge_capacity_commitments_rest_bad_request(transport: str='rest', request_type=reservation.MergeCapacityCommitmentsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.merge_capacity_commitments(request)

def test_merge_capacity_commitments_rest_flattened():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.CapacityCommitment()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', capacity_commitment_ids=['capacity_commitment_ids_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.CapacityCommitment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.merge_capacity_commitments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/capacityCommitments:merge' % client.transport._host, args[1])

def test_merge_capacity_commitments_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.merge_capacity_commitments(reservation.MergeCapacityCommitmentsRequest(), parent='parent_value', capacity_commitment_ids=['capacity_commitment_ids_value'])

def test_merge_capacity_commitments_rest_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.CreateAssignmentRequest, dict])
def test_create_assignment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
    request_init['assignment'] = {'name': 'name_value', 'assignee': 'assignee_value', 'job_type': 1, 'state': 1}
    test_field = reservation.CreateAssignmentRequest.meta.fields['assignment']

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
    for (field, value) in request_init['assignment'].items():
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
                for i in range(0, len(request_init['assignment'][field])):
                    del request_init['assignment'][field][i][subfield]
            else:
                del request_init['assignment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Assignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_assignment(request)
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

def test_create_assignment_rest_required_fields(request_type=reservation.CreateAssignmentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_assignment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('assignment_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.Assignment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.Assignment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_assignment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_assignment_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(('assignmentId',)) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_assignment_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_create_assignment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_create_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.CreateAssignmentRequest.pb(reservation.CreateAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.Assignment.to_json(reservation.Assignment())
        request = reservation.CreateAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.Assignment()
        client.create_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_assignment_rest_bad_request(transport: str='rest', request_type=reservation.CreateAssignmentRequest):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_assignment(request)

def test_create_assignment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Assignment()
        sample_request = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
        mock_args = dict(parent='parent_value', assignment=reservation.Assignment(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Assignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/reservations/*}/assignments' % client.transport._host, args[1])

def test_create_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_assignment(reservation.CreateAssignmentRequest(), parent='parent_value', assignment=reservation.Assignment(name='name_value'))

def test_create_assignment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.ListAssignmentsRequest, dict])
def test_list_assignments_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.ListAssignmentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.ListAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_assignments(request)
    assert isinstance(response, pagers.ListAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_assignments_rest_required_fields(request_type=reservation.ListAssignmentsRequest):
    if False:
        return 10
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_assignments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_assignments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.ListAssignmentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.ListAssignmentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_assignments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_assignments_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_assignments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_assignments_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_list_assignments') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_list_assignments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.ListAssignmentsRequest.pb(reservation.ListAssignmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.ListAssignmentsResponse.to_json(reservation.ListAssignmentsResponse())
        request = reservation.ListAssignmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.ListAssignmentsResponse()
        client.list_assignments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_assignments_rest_bad_request(transport: str='rest', request_type=reservation.ListAssignmentsRequest):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_assignments(request)

def test_list_assignments_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.ListAssignmentsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.ListAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_assignments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/reservations/*}/assignments' % client.transport._host, args[1])

def test_list_assignments_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_assignments(reservation.ListAssignmentsRequest(), parent='parent_value')

def test_list_assignments_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.ListAssignmentsResponse(assignments=[], next_page_token='def'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.ListAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]))
        response = response + response
        response = tuple((reservation.ListAssignmentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/reservations/sample3'}
        pager = client.list_assignments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Assignment) for i in results))
        pages = list(client.list_assignments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.DeleteAssignmentRequest, dict])
def test_delete_assignment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_assignment(request)
    assert response is None

def test_delete_assignment_rest_required_fields(request_type=reservation.DeleteAssignmentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_assignment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_assignment_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_assignment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_delete_assignment') as pre:
        pre.assert_not_called()
        pb_message = reservation.DeleteAssignmentRequest.pb(reservation.DeleteAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = reservation.DeleteAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_assignment_rest_bad_request(transport: str='rest', request_type=reservation.DeleteAssignmentRequest):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_assignment(request)

def test_delete_assignment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/reservations/*/assignments/*}' % client.transport._host, args[1])

def test_delete_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_assignment(reservation.DeleteAssignmentRequest(), name='name_value')

def test_delete_assignment_rest_error():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.SearchAssignmentsRequest, dict])
def test_search_assignments_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.SearchAssignmentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.SearchAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_assignments(request)
    assert isinstance(response, pagers.SearchAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_assignments_rest_required_fields(request_type=reservation.SearchAssignmentsRequest):
    if False:
        return 10
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_assignments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_assignments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'query'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.SearchAssignmentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.SearchAssignmentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_assignments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_assignments_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_assignments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'query')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_assignments_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_search_assignments') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_search_assignments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.SearchAssignmentsRequest.pb(reservation.SearchAssignmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.SearchAssignmentsResponse.to_json(reservation.SearchAssignmentsResponse())
        request = reservation.SearchAssignmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.SearchAssignmentsResponse()
        client.search_assignments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_assignments_rest_bad_request(transport: str='rest', request_type=reservation.SearchAssignmentsRequest):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_assignments(request)

def test_search_assignments_rest_flattened():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.SearchAssignmentsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', query='query_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.SearchAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_assignments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}:searchAssignments' % client.transport._host, args[1])

def test_search_assignments_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_assignments(reservation.SearchAssignmentsRequest(), parent='parent_value', query='query_value')

def test_search_assignments_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]))
        response = response + response
        response = tuple((reservation.SearchAssignmentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.search_assignments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Assignment) for i in results))
        pages = list(client.search_assignments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.SearchAllAssignmentsRequest, dict])
def test_search_all_assignments_rest(request_type):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.SearchAllAssignmentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.SearchAllAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_all_assignments(request)
    assert isinstance(response, pagers.SearchAllAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_all_assignments_rest_required_fields(request_type=reservation.SearchAllAssignmentsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_all_assignments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_all_assignments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'query'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.SearchAllAssignmentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.SearchAllAssignmentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_all_assignments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_all_assignments_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_all_assignments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'query')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_all_assignments_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_search_all_assignments') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_search_all_assignments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.SearchAllAssignmentsRequest.pb(reservation.SearchAllAssignmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.SearchAllAssignmentsResponse.to_json(reservation.SearchAllAssignmentsResponse())
        request = reservation.SearchAllAssignmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.SearchAllAssignmentsResponse()
        client.search_all_assignments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_all_assignments_rest_bad_request(transport: str='rest', request_type=reservation.SearchAllAssignmentsRequest):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_all_assignments(request)

def test_search_all_assignments_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.SearchAllAssignmentsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', query='query_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.SearchAllAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_all_assignments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}:searchAllAssignments' % client.transport._host, args[1])

def test_search_all_assignments_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_all_assignments(reservation.SearchAllAssignmentsRequest(), parent='parent_value', query='query_value')

def test_search_all_assignments_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment(), reservation.Assignment()], next_page_token='abc'), reservation.SearchAllAssignmentsResponse(assignments=[], next_page_token='def'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment()], next_page_token='ghi'), reservation.SearchAllAssignmentsResponse(assignments=[reservation.Assignment(), reservation.Assignment()]))
        response = response + response
        response = tuple((reservation.SearchAllAssignmentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.search_all_assignments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, reservation.Assignment) for i in results))
        pages = list(client.search_all_assignments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [reservation.MoveAssignmentRequest, dict])
def test_move_assignment_rest(request_type):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Assignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.move_assignment(request)
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

def test_move_assignment_rest_required_fields(request_type=reservation.MoveAssignmentRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).move_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).move_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.Assignment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.Assignment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.move_assignment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_move_assignment_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.move_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_move_assignment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_move_assignment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_move_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.MoveAssignmentRequest.pb(reservation.MoveAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.Assignment.to_json(reservation.Assignment())
        request = reservation.MoveAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.Assignment()
        client.move_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_move_assignment_rest_bad_request(transport: str='rest', request_type=reservation.MoveAssignmentRequest):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.move_assignment(request)

def test_move_assignment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Assignment()
        sample_request = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}
        mock_args = dict(name='name_value', destination_id='destination_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Assignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.move_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/reservations/*/assignments/*}:move' % client.transport._host, args[1])

def test_move_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.move_assignment(reservation.MoveAssignmentRequest(), name='name_value', destination_id='destination_id_value')

def test_move_assignment_rest_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.UpdateAssignmentRequest, dict])
def test_update_assignment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'assignment': {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}}
    request_init['assignment'] = {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4', 'assignee': 'assignee_value', 'job_type': 1, 'state': 1}
    test_field = reservation.UpdateAssignmentRequest.meta.fields['assignment']

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
    for (field, value) in request_init['assignment'].items():
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
                for i in range(0, len(request_init['assignment'][field])):
                    del request_init['assignment'][field][i][subfield]
            else:
                del request_init['assignment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Assignment(name='name_value', assignee='assignee_value', job_type=reservation.Assignment.JobType.PIPELINE, state=reservation.Assignment.State.PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Assignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_assignment(request)
    assert isinstance(response, reservation.Assignment)
    assert response.name == 'name_value'
    assert response.assignee == 'assignee_value'
    assert response.job_type == reservation.Assignment.JobType.PIPELINE
    assert response.state == reservation.Assignment.State.PENDING

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_assignment_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_update_assignment') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_update_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.UpdateAssignmentRequest.pb(reservation.UpdateAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.Assignment.to_json(reservation.Assignment())
        request = reservation.UpdateAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.Assignment()
        client.update_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_assignment_rest_bad_request(transport: str='rest', request_type=reservation.UpdateAssignmentRequest):
    if False:
        print('Hello World!')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'assignment': {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_assignment(request)

def test_update_assignment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.Assignment()
        sample_request = {'assignment': {'name': 'projects/sample1/locations/sample2/reservations/sample3/assignments/sample4'}}
        mock_args = dict(assignment=reservation.Assignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.Assignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{assignment.name=projects/*/locations/*/reservations/*/assignments/*}' % client.transport._host, args[1])

def test_update_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_assignment(reservation.UpdateAssignmentRequest(), assignment=reservation.Assignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_assignment_rest_error():
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.GetBiReservationRequest, dict])
def test_get_bi_reservation_rest(request_type):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/biReservation'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.BiReservation(name='name_value', size=443)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.BiReservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_bi_reservation(request)
    assert isinstance(response, reservation.BiReservation)
    assert response.name == 'name_value'
    assert response.size == 443

def test_get_bi_reservation_rest_required_fields(request_type=reservation.GetBiReservationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ReservationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_bi_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_bi_reservation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = reservation.BiReservation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = reservation.BiReservation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_bi_reservation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_bi_reservation_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_bi_reservation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_bi_reservation_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_get_bi_reservation') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_get_bi_reservation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.GetBiReservationRequest.pb(reservation.GetBiReservationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.BiReservation.to_json(reservation.BiReservation())
        request = reservation.GetBiReservationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.BiReservation()
        client.get_bi_reservation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_bi_reservation_rest_bad_request(transport: str='rest', request_type=reservation.GetBiReservationRequest):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/biReservation'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_bi_reservation(request)

def test_get_bi_reservation_rest_flattened():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.BiReservation()
        sample_request = {'name': 'projects/sample1/locations/sample2/biReservation'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.BiReservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_bi_reservation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/biReservation}' % client.transport._host, args[1])

def test_get_bi_reservation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_bi_reservation(reservation.GetBiReservationRequest(), name='name_value')

def test_get_bi_reservation_rest_error():
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [reservation.UpdateBiReservationRequest, dict])
def test_update_bi_reservation_rest(request_type):
    if False:
        while True:
            i = 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'bi_reservation': {'name': 'projects/sample1/locations/sample2/biReservation'}}
    request_init['bi_reservation'] = {'name': 'projects/sample1/locations/sample2/biReservation', 'update_time': {'seconds': 751, 'nanos': 543}, 'size': 443, 'preferred_tables': [{'project_id': 'project_id_value', 'dataset_id': 'dataset_id_value', 'table_id': 'table_id_value'}]}
    test_field = reservation.UpdateBiReservationRequest.meta.fields['bi_reservation']

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
    for (field, value) in request_init['bi_reservation'].items():
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
                for i in range(0, len(request_init['bi_reservation'][field])):
                    del request_init['bi_reservation'][field][i][subfield]
            else:
                del request_init['bi_reservation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.BiReservation(name='name_value', size=443)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.BiReservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_bi_reservation(request)
    assert isinstance(response, reservation.BiReservation)
    assert response.name == 'name_value'
    assert response.size == 443

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_bi_reservation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ReservationServiceRestInterceptor())
    client = ReservationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ReservationServiceRestInterceptor, 'post_update_bi_reservation') as post, mock.patch.object(transports.ReservationServiceRestInterceptor, 'pre_update_bi_reservation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = reservation.UpdateBiReservationRequest.pb(reservation.UpdateBiReservationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = reservation.BiReservation.to_json(reservation.BiReservation())
        request = reservation.UpdateBiReservationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = reservation.BiReservation()
        client.update_bi_reservation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_bi_reservation_rest_bad_request(transport: str='rest', request_type=reservation.UpdateBiReservationRequest):
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'bi_reservation': {'name': 'projects/sample1/locations/sample2/biReservation'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_bi_reservation(request)

def test_update_bi_reservation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = reservation.BiReservation()
        sample_request = {'bi_reservation': {'name': 'projects/sample1/locations/sample2/biReservation'}}
        mock_args = dict(bi_reservation=reservation.BiReservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = reservation.BiReservation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_bi_reservation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{bi_reservation.name=projects/*/locations/*/biReservation}' % client.transport._host, args[1])

def test_update_bi_reservation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_bi_reservation(reservation.UpdateBiReservationRequest(), bi_reservation=reservation.BiReservation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_bi_reservation_rest_error():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ReservationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ReservationServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ReservationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ReservationServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ReservationServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ReservationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ReservationServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.ReservationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ReservationServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.ReservationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ReservationServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ReservationServiceGrpcTransport, transports.ReservationServiceGrpcAsyncIOTransport, transports.ReservationServiceRestTransport])
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
        i = 10
        return i + 15
    transport = ReservationServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ReservationServiceGrpcTransport)

def test_reservation_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ReservationServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_reservation_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.bigquery_reservation_v1.services.reservation_service.transports.ReservationServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ReservationServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_reservation', 'list_reservations', 'get_reservation', 'delete_reservation', 'update_reservation', 'create_capacity_commitment', 'list_capacity_commitments', 'get_capacity_commitment', 'delete_capacity_commitment', 'update_capacity_commitment', 'split_capacity_commitment', 'merge_capacity_commitments', 'create_assignment', 'list_assignments', 'delete_assignment', 'search_assignments', 'search_all_assignments', 'move_assignment', 'update_assignment', 'get_bi_reservation', 'update_bi_reservation')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_reservation_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.bigquery_reservation_v1.services.reservation_service.transports.ReservationServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ReservationServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_reservation_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.bigquery_reservation_v1.services.reservation_service.transports.ReservationServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ReservationServiceTransport()
        adc.assert_called_once()

def test_reservation_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ReservationServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ReservationServiceGrpcTransport, transports.ReservationServiceGrpcAsyncIOTransport])
def test_reservation_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ReservationServiceGrpcTransport, transports.ReservationServiceGrpcAsyncIOTransport, transports.ReservationServiceRestTransport])
def test_reservation_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ReservationServiceGrpcTransport, grpc_helpers), (transports.ReservationServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_reservation_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('bigqueryreservation.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), scopes=['1', '2'], default_host='bigqueryreservation.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ReservationServiceGrpcTransport, transports.ReservationServiceGrpcAsyncIOTransport])
def test_reservation_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_reservation_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ReservationServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_reservation_service_host_no_port(transport_name):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='bigqueryreservation.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('bigqueryreservation.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryreservation.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_reservation_service_host_with_port(transport_name):
    if False:
        return 10
    client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='bigqueryreservation.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('bigqueryreservation.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryreservation.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_reservation_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ReservationServiceClient(credentials=creds1, transport=transport_name)
    client2 = ReservationServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_reservation._session
    session2 = client2.transport.create_reservation._session
    assert session1 != session2
    session1 = client1.transport.list_reservations._session
    session2 = client2.transport.list_reservations._session
    assert session1 != session2
    session1 = client1.transport.get_reservation._session
    session2 = client2.transport.get_reservation._session
    assert session1 != session2
    session1 = client1.transport.delete_reservation._session
    session2 = client2.transport.delete_reservation._session
    assert session1 != session2
    session1 = client1.transport.update_reservation._session
    session2 = client2.transport.update_reservation._session
    assert session1 != session2
    session1 = client1.transport.create_capacity_commitment._session
    session2 = client2.transport.create_capacity_commitment._session
    assert session1 != session2
    session1 = client1.transport.list_capacity_commitments._session
    session2 = client2.transport.list_capacity_commitments._session
    assert session1 != session2
    session1 = client1.transport.get_capacity_commitment._session
    session2 = client2.transport.get_capacity_commitment._session
    assert session1 != session2
    session1 = client1.transport.delete_capacity_commitment._session
    session2 = client2.transport.delete_capacity_commitment._session
    assert session1 != session2
    session1 = client1.transport.update_capacity_commitment._session
    session2 = client2.transport.update_capacity_commitment._session
    assert session1 != session2
    session1 = client1.transport.split_capacity_commitment._session
    session2 = client2.transport.split_capacity_commitment._session
    assert session1 != session2
    session1 = client1.transport.merge_capacity_commitments._session
    session2 = client2.transport.merge_capacity_commitments._session
    assert session1 != session2
    session1 = client1.transport.create_assignment._session
    session2 = client2.transport.create_assignment._session
    assert session1 != session2
    session1 = client1.transport.list_assignments._session
    session2 = client2.transport.list_assignments._session
    assert session1 != session2
    session1 = client1.transport.delete_assignment._session
    session2 = client2.transport.delete_assignment._session
    assert session1 != session2
    session1 = client1.transport.search_assignments._session
    session2 = client2.transport.search_assignments._session
    assert session1 != session2
    session1 = client1.transport.search_all_assignments._session
    session2 = client2.transport.search_all_assignments._session
    assert session1 != session2
    session1 = client1.transport.move_assignment._session
    session2 = client2.transport.move_assignment._session
    assert session1 != session2
    session1 = client1.transport.update_assignment._session
    session2 = client2.transport.update_assignment._session
    assert session1 != session2
    session1 = client1.transport.get_bi_reservation._session
    session2 = client2.transport.get_bi_reservation._session
    assert session1 != session2
    session1 = client1.transport.update_bi_reservation._session
    session2 = client2.transport.update_bi_reservation._session
    assert session1 != session2

def test_reservation_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ReservationServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_reservation_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ReservationServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ReservationServiceGrpcTransport, transports.ReservationServiceGrpcAsyncIOTransport])
def test_reservation_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class', [transports.ReservationServiceGrpcTransport, transports.ReservationServiceGrpcAsyncIOTransport])
def test_reservation_service_transport_channel_mtls_with_adc(transport_class):
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

def test_assignment_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    reservation = 'whelk'
    assignment = 'octopus'
    expected = 'projects/{project}/locations/{location}/reservations/{reservation}/assignments/{assignment}'.format(project=project, location=location, reservation=reservation, assignment=assignment)
    actual = ReservationServiceClient.assignment_path(project, location, reservation, assignment)
    assert expected == actual

def test_parse_assignment_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'reservation': 'cuttlefish', 'assignment': 'mussel'}
    path = ReservationServiceClient.assignment_path(**expected)
    actual = ReservationServiceClient.parse_assignment_path(path)
    assert expected == actual

def test_bi_reservation_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}/biReservation'.format(project=project, location=location)
    actual = ReservationServiceClient.bi_reservation_path(project, location)
    assert expected == actual

def test_parse_bi_reservation_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = ReservationServiceClient.bi_reservation_path(**expected)
    actual = ReservationServiceClient.parse_bi_reservation_path(path)
    assert expected == actual

def test_capacity_commitment_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    capacity_commitment = 'whelk'
    expected = 'projects/{project}/locations/{location}/capacityCommitments/{capacity_commitment}'.format(project=project, location=location, capacity_commitment=capacity_commitment)
    actual = ReservationServiceClient.capacity_commitment_path(project, location, capacity_commitment)
    assert expected == actual

def test_parse_capacity_commitment_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'capacity_commitment': 'nudibranch'}
    path = ReservationServiceClient.capacity_commitment_path(**expected)
    actual = ReservationServiceClient.parse_capacity_commitment_path(path)
    assert expected == actual

def test_reservation_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    reservation = 'winkle'
    expected = 'projects/{project}/locations/{location}/reservations/{reservation}'.format(project=project, location=location, reservation=reservation)
    actual = ReservationServiceClient.reservation_path(project, location, reservation)
    assert expected == actual

def test_parse_reservation_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus', 'location': 'scallop', 'reservation': 'abalone'}
    path = ReservationServiceClient.reservation_path(**expected)
    actual = ReservationServiceClient.parse_reservation_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ReservationServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'clam'}
    path = ReservationServiceClient.common_billing_account_path(**expected)
    actual = ReservationServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ReservationServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = ReservationServiceClient.common_folder_path(**expected)
    actual = ReservationServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ReservationServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = ReservationServiceClient.common_organization_path(**expected)
    actual = ReservationServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = ReservationServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel'}
    path = ReservationServiceClient.common_project_path(**expected)
    actual = ReservationServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ReservationServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = ReservationServiceClient.common_location_path(**expected)
    actual = ReservationServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ReservationServiceTransport, '_prep_wrapped_messages') as prep:
        client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ReservationServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ReservationServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ReservationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ReservationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ReservationServiceClient, transports.ReservationServiceGrpcTransport), (ReservationServiceAsyncClient, transports.ReservationServiceGrpcAsyncIOTransport)])
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