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
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.type import latlng_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.maps.fleetengine_v1.services.trip_service import TripServiceAsyncClient, TripServiceClient, pagers, transports
from google.maps.fleetengine_v1.types import fleetengine, header, traffic, trip_api, trips

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
    assert TripServiceClient._get_default_mtls_endpoint(None) is None
    assert TripServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TripServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TripServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TripServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TripServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TripServiceClient, 'grpc'), (TripServiceAsyncClient, 'grpc_asyncio')])
def test_trip_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'fleetengine.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TripServiceGrpcTransport, 'grpc'), (transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_trip_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TripServiceClient, 'grpc'), (TripServiceAsyncClient, 'grpc_asyncio')])
def test_trip_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'fleetengine.googleapis.com:443'

def test_trip_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = TripServiceClient.get_transport_class()
    available_transports = [transports.TripServiceGrpcTransport]
    assert transport in available_transports
    transport = TripServiceClient.get_transport_class('grpc')
    assert transport == transports.TripServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TripServiceClient, transports.TripServiceGrpcTransport, 'grpc'), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(TripServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TripServiceClient))
@mock.patch.object(TripServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TripServiceAsyncClient))
def test_trip_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(TripServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TripServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TripServiceClient, transports.TripServiceGrpcTransport, 'grpc', 'true'), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TripServiceClient, transports.TripServiceGrpcTransport, 'grpc', 'false'), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(TripServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TripServiceClient))
@mock.patch.object(TripServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TripServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_trip_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TripServiceClient, TripServiceAsyncClient])
@mock.patch.object(TripServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TripServiceClient))
@mock.patch.object(TripServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TripServiceAsyncClient))
def test_trip_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TripServiceClient, transports.TripServiceGrpcTransport, 'grpc'), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_trip_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TripServiceClient, transports.TripServiceGrpcTransport, 'grpc', grpc_helpers), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_trip_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_trip_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.maps.fleetengine_v1.services.trip_service.transports.TripServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TripServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TripServiceClient, transports.TripServiceGrpcTransport, 'grpc', grpc_helpers), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_trip_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('fleetengine.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='fleetengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [trip_api.CreateTripRequest, dict])
def test_create_trip(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_trip), '__call__') as call:
        call.return_value = trips.Trip(name='name_value', vehicle_id='vehicle_id_value', trip_status=trips.TripStatus.NEW, trip_type=fleetengine.TripType.SHARED, intermediate_destination_index=3187, current_route_segment='current_route_segment_value', number_of_passengers=2135, last_location_snappable=True, view=trips.TripView.SDK)
        response = client.create_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.CreateTripRequest()
    assert isinstance(response, trips.Trip)
    assert response.name == 'name_value'
    assert response.vehicle_id == 'vehicle_id_value'
    assert response.trip_status == trips.TripStatus.NEW
    assert response.trip_type == fleetengine.TripType.SHARED
    assert response.intermediate_destination_index == 3187
    assert response.current_route_segment == 'current_route_segment_value'
    assert response.number_of_passengers == 2135
    assert response.last_location_snappable is True
    assert response.view == trips.TripView.SDK

def test_create_trip_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_trip), '__call__') as call:
        client.create_trip()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.CreateTripRequest()

@pytest.mark.asyncio
async def test_create_trip_async(transport: str='grpc_asyncio', request_type=trip_api.CreateTripRequest):
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_trip), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trips.Trip(name='name_value', vehicle_id='vehicle_id_value', trip_status=trips.TripStatus.NEW, trip_type=fleetengine.TripType.SHARED, intermediate_destination_index=3187, current_route_segment='current_route_segment_value', number_of_passengers=2135, last_location_snappable=True, view=trips.TripView.SDK))
        response = await client.create_trip(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.CreateTripRequest()
    assert isinstance(response, trips.Trip)
    assert response.name == 'name_value'
    assert response.vehicle_id == 'vehicle_id_value'
    assert response.trip_status == trips.TripStatus.NEW
    assert response.trip_type == fleetengine.TripType.SHARED
    assert response.intermediate_destination_index == 3187
    assert response.current_route_segment == 'current_route_segment_value'
    assert response.number_of_passengers == 2135
    assert response.last_location_snappable is True
    assert response.view == trips.TripView.SDK

@pytest.mark.asyncio
async def test_create_trip_async_from_dict():
    await test_create_trip_async(request_type=dict)

def test_create_trip_routing_parameters():
    if False:
        for i in range(10):
            print('nop')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = trip_api.CreateTripRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.create_trip), '__call__') as call:
        call.return_value = trips.Trip()
        client.create_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

@pytest.mark.parametrize('request_type', [trip_api.GetTripRequest, dict])
def test_get_trip(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_trip), '__call__') as call:
        call.return_value = trips.Trip(name='name_value', vehicle_id='vehicle_id_value', trip_status=trips.TripStatus.NEW, trip_type=fleetengine.TripType.SHARED, intermediate_destination_index=3187, current_route_segment='current_route_segment_value', number_of_passengers=2135, last_location_snappable=True, view=trips.TripView.SDK)
        response = client.get_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.GetTripRequest()
    assert isinstance(response, trips.Trip)
    assert response.name == 'name_value'
    assert response.vehicle_id == 'vehicle_id_value'
    assert response.trip_status == trips.TripStatus.NEW
    assert response.trip_type == fleetengine.TripType.SHARED
    assert response.intermediate_destination_index == 3187
    assert response.current_route_segment == 'current_route_segment_value'
    assert response.number_of_passengers == 2135
    assert response.last_location_snappable is True
    assert response.view == trips.TripView.SDK

def test_get_trip_empty_call():
    if False:
        while True:
            i = 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_trip), '__call__') as call:
        client.get_trip()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.GetTripRequest()

@pytest.mark.asyncio
async def test_get_trip_async(transport: str='grpc_asyncio', request_type=trip_api.GetTripRequest):
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_trip), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trips.Trip(name='name_value', vehicle_id='vehicle_id_value', trip_status=trips.TripStatus.NEW, trip_type=fleetengine.TripType.SHARED, intermediate_destination_index=3187, current_route_segment='current_route_segment_value', number_of_passengers=2135, last_location_snappable=True, view=trips.TripView.SDK))
        response = await client.get_trip(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.GetTripRequest()
    assert isinstance(response, trips.Trip)
    assert response.name == 'name_value'
    assert response.vehicle_id == 'vehicle_id_value'
    assert response.trip_status == trips.TripStatus.NEW
    assert response.trip_type == fleetengine.TripType.SHARED
    assert response.intermediate_destination_index == 3187
    assert response.current_route_segment == 'current_route_segment_value'
    assert response.number_of_passengers == 2135
    assert response.last_location_snappable is True
    assert response.view == trips.TripView.SDK

@pytest.mark.asyncio
async def test_get_trip_async_from_dict():
    await test_get_trip_async(request_type=dict)

def test_get_trip_routing_parameters():
    if False:
        return 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = trip_api.GetTripRequest(**{'name': 'providers/sample1'})
    with mock.patch.object(type(client.transport.get_trip), '__call__') as call:
        call.return_value = trips.Trip()
        client.get_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

@pytest.mark.parametrize('request_type', [trip_api.ReportBillableTripRequest, dict])
def test_report_billable_trip(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.report_billable_trip), '__call__') as call:
        call.return_value = None
        response = client.report_billable_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.ReportBillableTripRequest()
    assert response is None

def test_report_billable_trip_empty_call():
    if False:
        return 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.report_billable_trip), '__call__') as call:
        client.report_billable_trip()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.ReportBillableTripRequest()

@pytest.mark.asyncio
async def test_report_billable_trip_async(transport: str='grpc_asyncio', request_type=trip_api.ReportBillableTripRequest):
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.report_billable_trip), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.report_billable_trip(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.ReportBillableTripRequest()
    assert response is None

@pytest.mark.asyncio
async def test_report_billable_trip_async_from_dict():
    await test_report_billable_trip_async(request_type=dict)

def test_report_billable_trip_routing_parameters():
    if False:
        for i in range(10):
            print('nop')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = trip_api.ReportBillableTripRequest(**{'name': 'providers/sample1'})
    with mock.patch.object(type(client.transport.report_billable_trip), '__call__') as call:
        call.return_value = None
        client.report_billable_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

@pytest.mark.parametrize('request_type', [trip_api.SearchTripsRequest, dict])
def test_search_trips(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_trips), '__call__') as call:
        call.return_value = trip_api.SearchTripsResponse(next_page_token='next_page_token_value')
        response = client.search_trips(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.SearchTripsRequest()
    assert isinstance(response, pagers.SearchTripsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_trips_empty_call():
    if False:
        i = 10
        return i + 15
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_trips), '__call__') as call:
        client.search_trips()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.SearchTripsRequest()

@pytest.mark.asyncio
async def test_search_trips_async(transport: str='grpc_asyncio', request_type=trip_api.SearchTripsRequest):
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_trips), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trip_api.SearchTripsResponse(next_page_token='next_page_token_value'))
        response = await client.search_trips(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.SearchTripsRequest()
    assert isinstance(response, pagers.SearchTripsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_trips_async_from_dict():
    await test_search_trips_async(request_type=dict)

def test_search_trips_routing_parameters():
    if False:
        i = 10
        return i + 15
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = trip_api.SearchTripsRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.search_trips), '__call__') as call:
        call.return_value = trip_api.SearchTripsResponse()
        client.search_trips(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_search_trips_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_trips), '__call__') as call:
        call.side_effect = (trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip(), trips.Trip()], next_page_token='abc'), trip_api.SearchTripsResponse(trips=[], next_page_token='def'), trip_api.SearchTripsResponse(trips=[trips.Trip()], next_page_token='ghi'), trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip()]), RuntimeError)
        metadata = ()
        pager = client.search_trips(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, trips.Trip) for i in results))

def test_search_trips_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_trips), '__call__') as call:
        call.side_effect = (trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip(), trips.Trip()], next_page_token='abc'), trip_api.SearchTripsResponse(trips=[], next_page_token='def'), trip_api.SearchTripsResponse(trips=[trips.Trip()], next_page_token='ghi'), trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip()]), RuntimeError)
        pages = list(client.search_trips(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_trips_async_pager():
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_trips), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip(), trips.Trip()], next_page_token='abc'), trip_api.SearchTripsResponse(trips=[], next_page_token='def'), trip_api.SearchTripsResponse(trips=[trips.Trip()], next_page_token='ghi'), trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip()]), RuntimeError)
        async_pager = await client.search_trips(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, trips.Trip) for i in responses))

@pytest.mark.asyncio
async def test_search_trips_async_pages():
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_trips), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip(), trips.Trip()], next_page_token='abc'), trip_api.SearchTripsResponse(trips=[], next_page_token='def'), trip_api.SearchTripsResponse(trips=[trips.Trip()], next_page_token='ghi'), trip_api.SearchTripsResponse(trips=[trips.Trip(), trips.Trip()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_trips(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [trip_api.UpdateTripRequest, dict])
def test_update_trip(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_trip), '__call__') as call:
        call.return_value = trips.Trip(name='name_value', vehicle_id='vehicle_id_value', trip_status=trips.TripStatus.NEW, trip_type=fleetengine.TripType.SHARED, intermediate_destination_index=3187, current_route_segment='current_route_segment_value', number_of_passengers=2135, last_location_snappable=True, view=trips.TripView.SDK)
        response = client.update_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.UpdateTripRequest()
    assert isinstance(response, trips.Trip)
    assert response.name == 'name_value'
    assert response.vehicle_id == 'vehicle_id_value'
    assert response.trip_status == trips.TripStatus.NEW
    assert response.trip_type == fleetengine.TripType.SHARED
    assert response.intermediate_destination_index == 3187
    assert response.current_route_segment == 'current_route_segment_value'
    assert response.number_of_passengers == 2135
    assert response.last_location_snappable is True
    assert response.view == trips.TripView.SDK

def test_update_trip_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_trip), '__call__') as call:
        client.update_trip()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.UpdateTripRequest()

@pytest.mark.asyncio
async def test_update_trip_async(transport: str='grpc_asyncio', request_type=trip_api.UpdateTripRequest):
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_trip), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trips.Trip(name='name_value', vehicle_id='vehicle_id_value', trip_status=trips.TripStatus.NEW, trip_type=fleetengine.TripType.SHARED, intermediate_destination_index=3187, current_route_segment='current_route_segment_value', number_of_passengers=2135, last_location_snappable=True, view=trips.TripView.SDK))
        response = await client.update_trip(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == trip_api.UpdateTripRequest()
    assert isinstance(response, trips.Trip)
    assert response.name == 'name_value'
    assert response.vehicle_id == 'vehicle_id_value'
    assert response.trip_status == trips.TripStatus.NEW
    assert response.trip_type == fleetengine.TripType.SHARED
    assert response.intermediate_destination_index == 3187
    assert response.current_route_segment == 'current_route_segment_value'
    assert response.number_of_passengers == 2135
    assert response.last_location_snappable is True
    assert response.view == trips.TripView.SDK

@pytest.mark.asyncio
async def test_update_trip_async_from_dict():
    await test_update_trip_async(request_type=dict)

def test_update_trip_routing_parameters():
    if False:
        while True:
            i = 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = trip_api.UpdateTripRequest(**{'name': 'providers/sample1'})
    with mock.patch.object(type(client.transport.update_trip), '__call__') as call:
        call.return_value = trips.Trip()
        client.update_trip(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.TripServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TripServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TripServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TripServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TripServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TripServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TripServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TripServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.TripServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TripServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.TripServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TripServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TripServiceGrpcTransport, transports.TripServiceGrpcAsyncIOTransport])
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
        return 10
    transport = TripServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TripServiceGrpcTransport)

def test_trip_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TripServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_trip_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.maps.fleetengine_v1.services.trip_service.transports.TripServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TripServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_trip', 'get_trip', 'report_billable_trip', 'search_trips', 'update_trip')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_trip_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.maps.fleetengine_v1.services.trip_service.transports.TripServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TripServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_trip_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.maps.fleetengine_v1.services.trip_service.transports.TripServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TripServiceTransport()
        adc.assert_called_once()

def test_trip_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TripServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TripServiceGrpcTransport, transports.TripServiceGrpcAsyncIOTransport])
def test_trip_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TripServiceGrpcTransport, transports.TripServiceGrpcAsyncIOTransport])
def test_trip_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TripServiceGrpcTransport, grpc_helpers), (transports.TripServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_trip_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('fleetengine.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='fleetengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TripServiceGrpcTransport, transports.TripServiceGrpcAsyncIOTransport])
def test_trip_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        i = 10
        return i + 15
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
def test_trip_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='fleetengine.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'fleetengine.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_trip_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='fleetengine.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'fleetengine.googleapis.com:8000'

def test_trip_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TripServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_trip_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TripServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TripServiceGrpcTransport, transports.TripServiceGrpcAsyncIOTransport])
def test_trip_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TripServiceGrpcTransport, transports.TripServiceGrpcAsyncIOTransport])
def test_trip_service_transport_channel_mtls_with_adc(transport_class):
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

def test_trip_path():
    if False:
        i = 10
        return i + 15
    provider = 'squid'
    trip = 'clam'
    expected = 'providers/{provider}/trips/{trip}'.format(provider=provider, trip=trip)
    actual = TripServiceClient.trip_path(provider, trip)
    assert expected == actual

def test_parse_trip_path():
    if False:
        while True:
            i = 10
    expected = {'provider': 'whelk', 'trip': 'octopus'}
    path = TripServiceClient.trip_path(**expected)
    actual = TripServiceClient.parse_trip_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TripServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nudibranch'}
    path = TripServiceClient.common_billing_account_path(**expected)
    actual = TripServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TripServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'mussel'}
    path = TripServiceClient.common_folder_path(**expected)
    actual = TripServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TripServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = TripServiceClient.common_organization_path(**expected)
    actual = TripServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = TripServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone'}
    path = TripServiceClient.common_project_path(**expected)
    actual = TripServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TripServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = TripServiceClient.common_location_path(**expected)
    actual = TripServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TripServiceTransport, '_prep_wrapped_messages') as prep:
        client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TripServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = TripServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TripServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        while True:
            i = 10
    transports = ['grpc']
    for transport in transports:
        client = TripServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TripServiceClient, transports.TripServiceGrpcTransport), (TripServiceAsyncClient, transports.TripServiceGrpcAsyncIOTransport)])
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