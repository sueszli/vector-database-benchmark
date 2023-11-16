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
from google.geo.type.types import viewport
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.type import latlng_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.maps.fleetengine_delivery_v1.services.delivery_service import DeliveryServiceAsyncClient, DeliveryServiceClient, pagers, transports
from google.maps.fleetengine_delivery_v1.types import common, delivery_api, delivery_vehicles, header, task_tracking_info, tasks

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert DeliveryServiceClient._get_default_mtls_endpoint(None) is None
    assert DeliveryServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DeliveryServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DeliveryServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DeliveryServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DeliveryServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DeliveryServiceClient, 'grpc'), (DeliveryServiceAsyncClient, 'grpc_asyncio'), (DeliveryServiceClient, 'rest')])
def test_delivery_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('fleetengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://fleetengine.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DeliveryServiceGrpcTransport, 'grpc'), (transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.DeliveryServiceRestTransport, 'rest')])
def test_delivery_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DeliveryServiceClient, 'grpc'), (DeliveryServiceAsyncClient, 'grpc_asyncio'), (DeliveryServiceClient, 'rest')])
def test_delivery_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('fleetengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://fleetengine.googleapis.com')

def test_delivery_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = DeliveryServiceClient.get_transport_class()
    available_transports = [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceRestTransport]
    assert transport in available_transports
    transport = DeliveryServiceClient.get_transport_class('grpc')
    assert transport == transports.DeliveryServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DeliveryServiceClient, transports.DeliveryServiceGrpcTransport, 'grpc'), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DeliveryServiceClient, transports.DeliveryServiceRestTransport, 'rest')])
@mock.patch.object(DeliveryServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DeliveryServiceClient))
@mock.patch.object(DeliveryServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DeliveryServiceAsyncClient))
def test_delivery_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(DeliveryServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DeliveryServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DeliveryServiceClient, transports.DeliveryServiceGrpcTransport, 'grpc', 'true'), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DeliveryServiceClient, transports.DeliveryServiceGrpcTransport, 'grpc', 'false'), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (DeliveryServiceClient, transports.DeliveryServiceRestTransport, 'rest', 'true'), (DeliveryServiceClient, transports.DeliveryServiceRestTransport, 'rest', 'false')])
@mock.patch.object(DeliveryServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DeliveryServiceClient))
@mock.patch.object(DeliveryServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DeliveryServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_delivery_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('client_class', [DeliveryServiceClient, DeliveryServiceAsyncClient])
@mock.patch.object(DeliveryServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DeliveryServiceClient))
@mock.patch.object(DeliveryServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DeliveryServiceAsyncClient))
def test_delivery_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DeliveryServiceClient, transports.DeliveryServiceGrpcTransport, 'grpc'), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DeliveryServiceClient, transports.DeliveryServiceRestTransport, 'rest')])
def test_delivery_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DeliveryServiceClient, transports.DeliveryServiceGrpcTransport, 'grpc', grpc_helpers), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (DeliveryServiceClient, transports.DeliveryServiceRestTransport, 'rest', None)])
def test_delivery_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_delivery_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.maps.fleetengine_delivery_v1.services.delivery_service.transports.DeliveryServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DeliveryServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DeliveryServiceClient, transports.DeliveryServiceGrpcTransport, 'grpc', grpc_helpers), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_delivery_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('fleetengine.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='fleetengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [delivery_api.CreateDeliveryVehicleRequest, dict])
def test_create_delivery_vehicle(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO)
        response = client.create_delivery_vehicle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.CreateDeliveryVehicleRequest()
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

def test_create_delivery_vehicle_empty_call():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_delivery_vehicle), '__call__') as call:
        client.create_delivery_vehicle()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.CreateDeliveryVehicleRequest()

@pytest.mark.asyncio
async def test_create_delivery_vehicle_async(transport: str='grpc_asyncio', request_type=delivery_api.CreateDeliveryVehicleRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_delivery_vehicle), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO))
        response = await client.create_delivery_vehicle(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.CreateDeliveryVehicleRequest()
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

@pytest.mark.asyncio
async def test_create_delivery_vehicle_async_from_dict():
    await test_create_delivery_vehicle_async(request_type=dict)

def test_create_delivery_vehicle_routing_parameters():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.CreateDeliveryVehicleRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.create_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        client.create_delivery_vehicle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_create_delivery_vehicle_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        client.create_delivery_vehicle(parent='parent_value', delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), delivery_vehicle_id='delivery_vehicle_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].delivery_vehicle
        mock_val = delivery_vehicles.DeliveryVehicle(name='name_value')
        assert arg == mock_val
        arg = args[0].delivery_vehicle_id
        mock_val = 'delivery_vehicle_id_value'
        assert arg == mock_val

def test_create_delivery_vehicle_flattened_error():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_delivery_vehicle(delivery_api.CreateDeliveryVehicleRequest(), parent='parent_value', delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), delivery_vehicle_id='delivery_vehicle_id_value')

@pytest.mark.asyncio
async def test_create_delivery_vehicle_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_vehicles.DeliveryVehicle())
        response = await client.create_delivery_vehicle(parent='parent_value', delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), delivery_vehicle_id='delivery_vehicle_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].delivery_vehicle
        mock_val = delivery_vehicles.DeliveryVehicle(name='name_value')
        assert arg == mock_val
        arg = args[0].delivery_vehicle_id
        mock_val = 'delivery_vehicle_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_delivery_vehicle_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_delivery_vehicle(delivery_api.CreateDeliveryVehicleRequest(), parent='parent_value', delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), delivery_vehicle_id='delivery_vehicle_id_value')

@pytest.mark.parametrize('request_type', [delivery_api.GetDeliveryVehicleRequest, dict])
def test_get_delivery_vehicle(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO)
        response = client.get_delivery_vehicle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetDeliveryVehicleRequest()
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

def test_get_delivery_vehicle_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_delivery_vehicle), '__call__') as call:
        client.get_delivery_vehicle()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetDeliveryVehicleRequest()

@pytest.mark.asyncio
async def test_get_delivery_vehicle_async(transport: str='grpc_asyncio', request_type=delivery_api.GetDeliveryVehicleRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_delivery_vehicle), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO))
        response = await client.get_delivery_vehicle(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetDeliveryVehicleRequest()
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

@pytest.mark.asyncio
async def test_get_delivery_vehicle_async_from_dict():
    await test_get_delivery_vehicle_async(request_type=dict)

def test_get_delivery_vehicle_routing_parameters():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.GetDeliveryVehicleRequest(**{'name': 'providers/sample1'})
    with mock.patch.object(type(client.transport.get_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        client.get_delivery_vehicle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_get_delivery_vehicle_flattened():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        client.get_delivery_vehicle(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_delivery_vehicle_flattened_error():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_delivery_vehicle(delivery_api.GetDeliveryVehicleRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_delivery_vehicle_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_vehicles.DeliveryVehicle())
        response = await client.get_delivery_vehicle(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_delivery_vehicle_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_delivery_vehicle(delivery_api.GetDeliveryVehicleRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [delivery_api.UpdateDeliveryVehicleRequest, dict])
def test_update_delivery_vehicle(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO)
        response = client.update_delivery_vehicle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.UpdateDeliveryVehicleRequest()
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

def test_update_delivery_vehicle_empty_call():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_delivery_vehicle), '__call__') as call:
        client.update_delivery_vehicle()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.UpdateDeliveryVehicleRequest()

@pytest.mark.asyncio
async def test_update_delivery_vehicle_async(transport: str='grpc_asyncio', request_type=delivery_api.UpdateDeliveryVehicleRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_delivery_vehicle), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO))
        response = await client.update_delivery_vehicle(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.UpdateDeliveryVehicleRequest()
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

@pytest.mark.asyncio
async def test_update_delivery_vehicle_async_from_dict():
    await test_update_delivery_vehicle_async(request_type=dict)

def test_update_delivery_vehicle_routing_parameters():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.UpdateDeliveryVehicleRequest(**{'delivery_vehicle': {'name': 'providers/sample1'}})
    with mock.patch.object(type(client.transport.update_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        client.update_delivery_vehicle(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_update_delivery_vehicle_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        client.update_delivery_vehicle(delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].delivery_vehicle
        mock_val = delivery_vehicles.DeliveryVehicle(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_delivery_vehicle_flattened_error():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_delivery_vehicle(delivery_api.UpdateDeliveryVehicleRequest(), delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_delivery_vehicle_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_delivery_vehicle), '__call__') as call:
        call.return_value = delivery_vehicles.DeliveryVehicle()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_vehicles.DeliveryVehicle())
        response = await client.update_delivery_vehicle(delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].delivery_vehicle
        mock_val = delivery_vehicles.DeliveryVehicle(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_delivery_vehicle_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_delivery_vehicle(delivery_api.UpdateDeliveryVehicleRequest(), delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [delivery_api.BatchCreateTasksRequest, dict])
def test_batch_create_tasks(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_tasks), '__call__') as call:
        call.return_value = delivery_api.BatchCreateTasksResponse()
        response = client.batch_create_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.BatchCreateTasksRequest()
    assert isinstance(response, delivery_api.BatchCreateTasksResponse)

def test_batch_create_tasks_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_tasks), '__call__') as call:
        client.batch_create_tasks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.BatchCreateTasksRequest()

@pytest.mark.asyncio
async def test_batch_create_tasks_async(transport: str='grpc_asyncio', request_type=delivery_api.BatchCreateTasksRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_tasks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.BatchCreateTasksResponse())
        response = await client.batch_create_tasks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.BatchCreateTasksRequest()
    assert isinstance(response, delivery_api.BatchCreateTasksResponse)

@pytest.mark.asyncio
async def test_batch_create_tasks_async_from_dict():
    await test_batch_create_tasks_async(request_type=dict)

def test_batch_create_tasks_routing_parameters():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.BatchCreateTasksRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.batch_create_tasks), '__call__') as call:
        call.return_value = delivery_api.BatchCreateTasksResponse()
        client.batch_create_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

@pytest.mark.parametrize('request_type', [delivery_api.CreateTaskRequest, dict])
def test_create_task(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value')
        response = client.create_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.CreateTaskRequest()
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

def test_create_task_empty_call():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        client.create_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.CreateTaskRequest()

@pytest.mark.asyncio
async def test_create_task_async(transport: str='grpc_asyncio', request_type=delivery_api.CreateTaskRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value'))
        response = await client.create_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.CreateTaskRequest()
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

@pytest.mark.asyncio
async def test_create_task_async_from_dict():
    await test_create_task_async(request_type=dict)

def test_create_task_routing_parameters():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.CreateTaskRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = tasks.Task()
        client.create_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_create_task_flattened():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = tasks.Task()
        client.create_task(parent='parent_value', task=tasks.Task(name='name_value'), task_id='task_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].task
        mock_val = tasks.Task(name='name_value')
        assert arg == mock_val
        arg = args[0].task_id
        mock_val = 'task_id_value'
        assert arg == mock_val

def test_create_task_flattened_error():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_task(delivery_api.CreateTaskRequest(), parent='parent_value', task=tasks.Task(name='name_value'), task_id='task_id_value')

@pytest.mark.asyncio
async def test_create_task_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = tasks.Task()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tasks.Task())
        response = await client.create_task(parent='parent_value', task=tasks.Task(name='name_value'), task_id='task_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].task
        mock_val = tasks.Task(name='name_value')
        assert arg == mock_val
        arg = args[0].task_id
        mock_val = 'task_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_task_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_task(delivery_api.CreateTaskRequest(), parent='parent_value', task=tasks.Task(name='name_value'), task_id='task_id_value')

@pytest.mark.parametrize('request_type', [delivery_api.GetTaskRequest, dict])
def test_get_task(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value')
        response = client.get_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetTaskRequest()
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

def test_get_task_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        client.get_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetTaskRequest()

@pytest.mark.asyncio
async def test_get_task_async(transport: str='grpc_asyncio', request_type=delivery_api.GetTaskRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value'))
        response = await client.get_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetTaskRequest()
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

@pytest.mark.asyncio
async def test_get_task_async_from_dict():
    await test_get_task_async(request_type=dict)

def test_get_task_routing_parameters():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.GetTaskRequest(**{'name': 'providers/sample1'})
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = tasks.Task()
        client.get_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_get_task_flattened():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = tasks.Task()
        client.get_task(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_task_flattened_error():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_task(delivery_api.GetTaskRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_task_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = tasks.Task()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tasks.Task())
        response = await client.get_task(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_task_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_task(delivery_api.GetTaskRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [delivery_api.SearchTasksRequest, dict])
def test_search_tasks(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.return_value = delivery_api.SearchTasksResponse(next_page_token='next_page_token_value')
        response = client.search_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.SearchTasksRequest()
    assert isinstance(response, pagers.SearchTasksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_tasks_empty_call():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        client.search_tasks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.SearchTasksRequest()

@pytest.mark.asyncio
async def test_search_tasks_async(transport: str='grpc_asyncio', request_type=delivery_api.SearchTasksRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.SearchTasksResponse(next_page_token='next_page_token_value'))
        response = await client.search_tasks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.SearchTasksRequest()
    assert isinstance(response, pagers.SearchTasksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_tasks_async_from_dict():
    await test_search_tasks_async(request_type=dict)

def test_search_tasks_routing_parameters():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.SearchTasksRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.return_value = delivery_api.SearchTasksResponse()
        client.search_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_search_tasks_flattened():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.return_value = delivery_api.SearchTasksResponse()
        client.search_tasks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_search_tasks_flattened_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_tasks(delivery_api.SearchTasksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_search_tasks_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.return_value = delivery_api.SearchTasksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.SearchTasksResponse())
        response = await client.search_tasks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_tasks_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_tasks(delivery_api.SearchTasksRequest(), parent='parent_value')

def test_search_tasks_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.side_effect = (delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.SearchTasksResponse(tasks=[], next_page_token='def'), delivery_api.SearchTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        metadata = ()
        pager = client.search_tasks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tasks.Task) for i in results))

def test_search_tasks_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_tasks), '__call__') as call:
        call.side_effect = (delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.SearchTasksResponse(tasks=[], next_page_token='def'), delivery_api.SearchTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        pages = list(client.search_tasks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_tasks_async_pager():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_tasks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.SearchTasksResponse(tasks=[], next_page_token='def'), delivery_api.SearchTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        async_pager = await client.search_tasks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tasks.Task) for i in responses))

@pytest.mark.asyncio
async def test_search_tasks_async_pages():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_tasks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.SearchTasksResponse(tasks=[], next_page_token='def'), delivery_api.SearchTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_tasks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [delivery_api.UpdateTaskRequest, dict])
def test_update_task(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_task), '__call__') as call:
        call.return_value = tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value')
        response = client.update_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.UpdateTaskRequest()
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

def test_update_task_empty_call():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_task), '__call__') as call:
        client.update_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.UpdateTaskRequest()

@pytest.mark.asyncio
async def test_update_task_async(transport: str='grpc_asyncio', request_type=delivery_api.UpdateTaskRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value'))
        response = await client.update_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.UpdateTaskRequest()
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

@pytest.mark.asyncio
async def test_update_task_async_from_dict():
    await test_update_task_async(request_type=dict)

def test_update_task_routing_parameters():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.UpdateTaskRequest(**{'task': {'name': 'providers/sample1'}})
    with mock.patch.object(type(client.transport.update_task), '__call__') as call:
        call.return_value = tasks.Task()
        client.update_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_update_task_flattened():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_task), '__call__') as call:
        call.return_value = tasks.Task()
        client.update_task(task=tasks.Task(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].task
        mock_val = tasks.Task(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_task_flattened_error():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_task(delivery_api.UpdateTaskRequest(), task=tasks.Task(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_task_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_task), '__call__') as call:
        call.return_value = tasks.Task()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tasks.Task())
        response = await client.update_task(task=tasks.Task(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].task
        mock_val = tasks.Task(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_task_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_task(delivery_api.UpdateTaskRequest(), task=tasks.Task(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [delivery_api.ListTasksRequest, dict])
def test_list_tasks(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = delivery_api.ListTasksResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.ListTasksRequest()
    assert isinstance(response, pagers.ListTasksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_tasks_empty_call():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        client.list_tasks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.ListTasksRequest()

@pytest.mark.asyncio
async def test_list_tasks_async(transport: str='grpc_asyncio', request_type=delivery_api.ListTasksRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.ListTasksResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_tasks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.ListTasksRequest()
    assert isinstance(response, pagers.ListTasksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_tasks_async_from_dict():
    await test_list_tasks_async(request_type=dict)

def test_list_tasks_routing_parameters():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.ListTasksRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = delivery_api.ListTasksResponse()
        client.list_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_list_tasks_flattened():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = delivery_api.ListTasksResponse()
        client.list_tasks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tasks_flattened_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tasks(delivery_api.ListTasksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tasks_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = delivery_api.ListTasksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.ListTasksResponse())
        response = await client.list_tasks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tasks_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tasks(delivery_api.ListTasksRequest(), parent='parent_value')

def test_list_tasks_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.side_effect = (delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.ListTasksResponse(tasks=[], next_page_token='def'), delivery_api.ListTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        metadata = ()
        pager = client.list_tasks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tasks.Task) for i in results))

def test_list_tasks_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.side_effect = (delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.ListTasksResponse(tasks=[], next_page_token='def'), delivery_api.ListTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        pages = list(client.list_tasks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tasks_async_pager():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tasks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.ListTasksResponse(tasks=[], next_page_token='def'), delivery_api.ListTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        async_pager = await client.list_tasks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tasks.Task) for i in responses))

@pytest.mark.asyncio
async def test_list_tasks_async_pages():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tasks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.ListTasksResponse(tasks=[], next_page_token='def'), delivery_api.ListTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tasks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [delivery_api.GetTaskTrackingInfoRequest, dict])
def test_get_task_tracking_info(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_task_tracking_info), '__call__') as call:
        call.return_value = task_tracking_info.TaskTrackingInfo(name='name_value', tracking_id='tracking_id_value', state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED)
        response = client.get_task_tracking_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetTaskTrackingInfoRequest()
    assert isinstance(response, task_tracking_info.TaskTrackingInfo)
    assert response.name == 'name_value'
    assert response.tracking_id == 'tracking_id_value'
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED

def test_get_task_tracking_info_empty_call():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_task_tracking_info), '__call__') as call:
        client.get_task_tracking_info()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetTaskTrackingInfoRequest()

@pytest.mark.asyncio
async def test_get_task_tracking_info_async(transport: str='grpc_asyncio', request_type=delivery_api.GetTaskTrackingInfoRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_task_tracking_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task_tracking_info.TaskTrackingInfo(name='name_value', tracking_id='tracking_id_value', state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED))
        response = await client.get_task_tracking_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.GetTaskTrackingInfoRequest()
    assert isinstance(response, task_tracking_info.TaskTrackingInfo)
    assert response.name == 'name_value'
    assert response.tracking_id == 'tracking_id_value'
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED

@pytest.mark.asyncio
async def test_get_task_tracking_info_async_from_dict():
    await test_get_task_tracking_info_async(request_type=dict)

def test_get_task_tracking_info_routing_parameters():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.GetTaskTrackingInfoRequest(**{'name': 'providers/sample1'})
    with mock.patch.object(type(client.transport.get_task_tracking_info), '__call__') as call:
        call.return_value = task_tracking_info.TaskTrackingInfo()
        client.get_task_tracking_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_get_task_tracking_info_flattened():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_task_tracking_info), '__call__') as call:
        call.return_value = task_tracking_info.TaskTrackingInfo()
        client.get_task_tracking_info(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_task_tracking_info_flattened_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_task_tracking_info(delivery_api.GetTaskTrackingInfoRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_task_tracking_info_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_task_tracking_info), '__call__') as call:
        call.return_value = task_tracking_info.TaskTrackingInfo()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task_tracking_info.TaskTrackingInfo())
        response = await client.get_task_tracking_info(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_task_tracking_info_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_task_tracking_info(delivery_api.GetTaskTrackingInfoRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [delivery_api.ListDeliveryVehiclesRequest, dict])
def test_list_delivery_vehicles(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.return_value = delivery_api.ListDeliveryVehiclesResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_delivery_vehicles(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.ListDeliveryVehiclesRequest()
    assert isinstance(response, pagers.ListDeliveryVehiclesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_delivery_vehicles_empty_call():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        client.list_delivery_vehicles()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.ListDeliveryVehiclesRequest()

@pytest.mark.asyncio
async def test_list_delivery_vehicles_async(transport: str='grpc_asyncio', request_type=delivery_api.ListDeliveryVehiclesRequest):
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.ListDeliveryVehiclesResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_delivery_vehicles(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == delivery_api.ListDeliveryVehiclesRequest()
    assert isinstance(response, pagers.ListDeliveryVehiclesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_delivery_vehicles_async_from_dict():
    await test_list_delivery_vehicles_async(request_type=dict)

def test_list_delivery_vehicles_routing_parameters():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = delivery_api.ListDeliveryVehiclesRequest(**{'parent': 'providers/sample1'})
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.return_value = delivery_api.ListDeliveryVehiclesResponse()
        client.list_delivery_vehicles(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert kw['metadata']

def test_list_delivery_vehicles_flattened():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.return_value = delivery_api.ListDeliveryVehiclesResponse()
        client.list_delivery_vehicles(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_delivery_vehicles_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_delivery_vehicles(delivery_api.ListDeliveryVehiclesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_delivery_vehicles_flattened_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.return_value = delivery_api.ListDeliveryVehiclesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(delivery_api.ListDeliveryVehiclesResponse())
        response = await client.list_delivery_vehicles(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_delivery_vehicles_flattened_error_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_delivery_vehicles(delivery_api.ListDeliveryVehiclesRequest(), parent='parent_value')

def test_list_delivery_vehicles_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.side_effect = (delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()], next_page_token='abc'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[], next_page_token='def'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle()], next_page_token='ghi'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()]), RuntimeError)
        metadata = ()
        pager = client.list_delivery_vehicles(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, delivery_vehicles.DeliveryVehicle) for i in results))

def test_list_delivery_vehicles_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__') as call:
        call.side_effect = (delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()], next_page_token='abc'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[], next_page_token='def'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle()], next_page_token='ghi'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()]), RuntimeError)
        pages = list(client.list_delivery_vehicles(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_delivery_vehicles_async_pager():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()], next_page_token='abc'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[], next_page_token='def'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle()], next_page_token='ghi'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()]), RuntimeError)
        async_pager = await client.list_delivery_vehicles(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, delivery_vehicles.DeliveryVehicle) for i in responses))

@pytest.mark.asyncio
async def test_list_delivery_vehicles_async_pages():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_delivery_vehicles), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()], next_page_token='abc'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[], next_page_token='def'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle()], next_page_token='ghi'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_delivery_vehicles(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [delivery_api.CreateDeliveryVehicleRequest, dict])
def test_create_delivery_vehicle_rest(request_type):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'providers/sample1'}
    request_init['delivery_vehicle'] = {'name': 'name_value', 'last_location': {'location': {'latitude': 0.86, 'longitude': 0.971}, 'horizontal_accuracy': {'value': 0.541}, 'latlng_accuracy': {}, 'heading': {'value': 541}, 'bearing_accuracy': {}, 'heading_accuracy': {}, 'altitude': {}, 'vertical_accuracy': {}, 'altitude_accuracy': {}, 'speed_kmph': {}, 'speed': {}, 'speed_accuracy': {}, 'update_time': {'seconds': 751, 'nanos': 543}, 'server_time': {}, 'location_sensor': 1, 'is_road_snapped': {'value': True}, 'is_gps_sensor_enabled': {}, 'time_since_update': {}, 'num_stale_updates': {}, 'raw_location': {}, 'raw_location_time': {}, 'raw_location_sensor': 1, 'raw_location_accuracy': {}, 'supplemental_location': {}, 'supplemental_location_time': {}, 'supplemental_location_sensor': 1, 'supplemental_location_accuracy': {}, 'road_snapped': True}, 'navigation_status': 1, 'current_route_segment': b'current_route_segment_blob', 'current_route_segment_end_point': {}, 'remaining_distance_meters': {}, 'remaining_duration': {'seconds': 751, 'nanos': 543}, 'remaining_vehicle_journey_segments': [{'stop': {'planned_location': {'point': {}}, 'tasks': [{'task_id': 'task_id_value', 'task_duration': {}, 'target_time_window': {'start_time': {}, 'end_time': {}}}], 'state': 1}, 'driving_distance_meters': {}, 'driving_duration': {}, 'path': {}}], 'attributes': [{'key': 'key_value', 'value': 'value_value', 'string_value': 'string_value_value', 'bool_value': True, 'number_value': 0.1285}], 'type_': 1}
    test_field = delivery_api.CreateDeliveryVehicleRequest.meta.fields['delivery_vehicle']

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
    for (field, value) in request_init['delivery_vehicle'].items():
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
                for i in range(0, len(request_init['delivery_vehicle'][field])):
                    del request_init['delivery_vehicle'][field][i][subfield]
            else:
                del request_init['delivery_vehicle'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_delivery_vehicle(request)
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

def test_create_delivery_vehicle_rest_required_fields(request_type=delivery_api.CreateDeliveryVehicleRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['delivery_vehicle_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'deliveryVehicleId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_delivery_vehicle._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'deliveryVehicleId' in jsonified_request
    assert jsonified_request['deliveryVehicleId'] == request_init['delivery_vehicle_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['deliveryVehicleId'] = 'delivery_vehicle_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_delivery_vehicle._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('delivery_vehicle_id', 'header'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'deliveryVehicleId' in jsonified_request
    assert jsonified_request['deliveryVehicleId'] == 'delivery_vehicle_id_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_vehicles.DeliveryVehicle()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_delivery_vehicle(request)
            expected_params = [('deliveryVehicleId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_delivery_vehicle_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_delivery_vehicle._get_unset_required_fields({})
    assert set(unset_fields) == set(('deliveryVehicleId', 'header')) & set(('parent', 'deliveryVehicleId', 'deliveryVehicle'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_delivery_vehicle_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_create_delivery_vehicle') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_create_delivery_vehicle') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.CreateDeliveryVehicleRequest.pb(delivery_api.CreateDeliveryVehicleRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_vehicles.DeliveryVehicle.to_json(delivery_vehicles.DeliveryVehicle())
        request = delivery_api.CreateDeliveryVehicleRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_vehicles.DeliveryVehicle()
        client.create_delivery_vehicle(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_delivery_vehicle_rest_bad_request(transport: str='rest', request_type=delivery_api.CreateDeliveryVehicleRequest):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_delivery_vehicle(request)

def test_create_delivery_vehicle_rest_flattened():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_vehicles.DeliveryVehicle()
        sample_request = {'parent': 'providers/sample1'}
        mock_args = dict(parent='parent_value', delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), delivery_vehicle_id='delivery_vehicle_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_delivery_vehicle(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=providers/*}/deliveryVehicles' % client.transport._host, args[1])

def test_create_delivery_vehicle_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_delivery_vehicle(delivery_api.CreateDeliveryVehicleRequest(), parent='parent_value', delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), delivery_vehicle_id='delivery_vehicle_id_value')

def test_create_delivery_vehicle_rest_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.GetDeliveryVehicleRequest, dict])
def test_get_delivery_vehicle_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'providers/sample1/deliveryVehicles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_delivery_vehicle(request)
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

def test_get_delivery_vehicle_rest_required_fields(request_type=delivery_api.GetDeliveryVehicleRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_delivery_vehicle._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_delivery_vehicle._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_vehicles.DeliveryVehicle()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_delivery_vehicle(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_delivery_vehicle_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_delivery_vehicle._get_unset_required_fields({})
    assert set(unset_fields) == set(('header',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_delivery_vehicle_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_get_delivery_vehicle') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_get_delivery_vehicle') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.GetDeliveryVehicleRequest.pb(delivery_api.GetDeliveryVehicleRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_vehicles.DeliveryVehicle.to_json(delivery_vehicles.DeliveryVehicle())
        request = delivery_api.GetDeliveryVehicleRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_vehicles.DeliveryVehicle()
        client.get_delivery_vehicle(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_delivery_vehicle_rest_bad_request(transport: str='rest', request_type=delivery_api.GetDeliveryVehicleRequest):
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'providers/sample1/deliveryVehicles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_delivery_vehicle(request)

def test_get_delivery_vehicle_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_vehicles.DeliveryVehicle()
        sample_request = {'name': 'providers/sample1/deliveryVehicles/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_delivery_vehicle(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=providers/*/deliveryVehicles/*}' % client.transport._host, args[1])

def test_get_delivery_vehicle_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_delivery_vehicle(delivery_api.GetDeliveryVehicleRequest(), name='name_value')

def test_get_delivery_vehicle_rest_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.UpdateDeliveryVehicleRequest, dict])
def test_update_delivery_vehicle_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'delivery_vehicle': {'name': 'providers/sample1/deliveryVehicles/sample2'}}
    request_init['delivery_vehicle'] = {'name': 'providers/sample1/deliveryVehicles/sample2', 'last_location': {'location': {'latitude': 0.86, 'longitude': 0.971}, 'horizontal_accuracy': {'value': 0.541}, 'latlng_accuracy': {}, 'heading': {'value': 541}, 'bearing_accuracy': {}, 'heading_accuracy': {}, 'altitude': {}, 'vertical_accuracy': {}, 'altitude_accuracy': {}, 'speed_kmph': {}, 'speed': {}, 'speed_accuracy': {}, 'update_time': {'seconds': 751, 'nanos': 543}, 'server_time': {}, 'location_sensor': 1, 'is_road_snapped': {'value': True}, 'is_gps_sensor_enabled': {}, 'time_since_update': {}, 'num_stale_updates': {}, 'raw_location': {}, 'raw_location_time': {}, 'raw_location_sensor': 1, 'raw_location_accuracy': {}, 'supplemental_location': {}, 'supplemental_location_time': {}, 'supplemental_location_sensor': 1, 'supplemental_location_accuracy': {}, 'road_snapped': True}, 'navigation_status': 1, 'current_route_segment': b'current_route_segment_blob', 'current_route_segment_end_point': {}, 'remaining_distance_meters': {}, 'remaining_duration': {'seconds': 751, 'nanos': 543}, 'remaining_vehicle_journey_segments': [{'stop': {'planned_location': {'point': {}}, 'tasks': [{'task_id': 'task_id_value', 'task_duration': {}, 'target_time_window': {'start_time': {}, 'end_time': {}}}], 'state': 1}, 'driving_distance_meters': {}, 'driving_duration': {}, 'path': {}}], 'attributes': [{'key': 'key_value', 'value': 'value_value', 'string_value': 'string_value_value', 'bool_value': True, 'number_value': 0.1285}], 'type_': 1}
    test_field = delivery_api.UpdateDeliveryVehicleRequest.meta.fields['delivery_vehicle']

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
    for (field, value) in request_init['delivery_vehicle'].items():
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
                for i in range(0, len(request_init['delivery_vehicle'][field])):
                    del request_init['delivery_vehicle'][field][i][subfield]
            else:
                del request_init['delivery_vehicle'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_vehicles.DeliveryVehicle(name='name_value', navigation_status=common.DeliveryVehicleNavigationStatus.NO_GUIDANCE, current_route_segment=b'current_route_segment_blob', type_=delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_delivery_vehicle(request)
    assert isinstance(response, delivery_vehicles.DeliveryVehicle)
    assert response.name == 'name_value'
    assert response.navigation_status == common.DeliveryVehicleNavigationStatus.NO_GUIDANCE
    assert response.current_route_segment == b'current_route_segment_blob'
    assert response.type_ == delivery_vehicles.DeliveryVehicle.DeliveryVehicleType.AUTO

def test_update_delivery_vehicle_rest_required_fields(request_type=delivery_api.UpdateDeliveryVehicleRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_delivery_vehicle._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_delivery_vehicle._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_vehicles.DeliveryVehicle()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_delivery_vehicle(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_delivery_vehicle_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_delivery_vehicle._get_unset_required_fields({})
    assert set(unset_fields) == set(('header', 'updateMask')) & set(('deliveryVehicle', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_delivery_vehicle_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_update_delivery_vehicle') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_update_delivery_vehicle') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.UpdateDeliveryVehicleRequest.pb(delivery_api.UpdateDeliveryVehicleRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_vehicles.DeliveryVehicle.to_json(delivery_vehicles.DeliveryVehicle())
        request = delivery_api.UpdateDeliveryVehicleRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_vehicles.DeliveryVehicle()
        client.update_delivery_vehicle(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_delivery_vehicle_rest_bad_request(transport: str='rest', request_type=delivery_api.UpdateDeliveryVehicleRequest):
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'delivery_vehicle': {'name': 'providers/sample1/deliveryVehicles/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_delivery_vehicle(request)

def test_update_delivery_vehicle_rest_flattened():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_vehicles.DeliveryVehicle()
        sample_request = {'delivery_vehicle': {'name': 'providers/sample1/deliveryVehicles/sample2'}}
        mock_args = dict(delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_vehicles.DeliveryVehicle.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_delivery_vehicle(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{delivery_vehicle.name=providers/*/deliveryVehicles/*}' % client.transport._host, args[1])

def test_update_delivery_vehicle_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_delivery_vehicle(delivery_api.UpdateDeliveryVehicleRequest(), delivery_vehicle=delivery_vehicles.DeliveryVehicle(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_delivery_vehicle_rest_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.BatchCreateTasksRequest, dict])
def test_batch_create_tasks_rest(request_type):
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.BatchCreateTasksResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.BatchCreateTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_tasks(request)
    assert isinstance(response, delivery_api.BatchCreateTasksResponse)

def test_batch_create_tasks_rest_required_fields(request_type=delivery_api.BatchCreateTasksRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_tasks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_tasks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_api.BatchCreateTasksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_api.BatchCreateTasksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_create_tasks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_tasks_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_tasks._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'requests'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_tasks_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_batch_create_tasks') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_batch_create_tasks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.BatchCreateTasksRequest.pb(delivery_api.BatchCreateTasksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_api.BatchCreateTasksResponse.to_json(delivery_api.BatchCreateTasksResponse())
        request = delivery_api.BatchCreateTasksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_api.BatchCreateTasksResponse()
        client.batch_create_tasks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_tasks_rest_bad_request(transport: str='rest', request_type=delivery_api.BatchCreateTasksRequest):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_tasks(request)

def test_batch_create_tasks_rest_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.CreateTaskRequest, dict])
def test_create_task_rest(request_type):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'providers/sample1'}
    request_init['task'] = {'name': 'name_value', 'type_': 1, 'state': 1, 'task_outcome': 1, 'task_outcome_time': {'seconds': 751, 'nanos': 543}, 'task_outcome_location': {'point': {'latitude': 0.86, 'longitude': 0.971}}, 'task_outcome_location_source': 2, 'tracking_id': 'tracking_id_value', 'delivery_vehicle_id': 'delivery_vehicle_id_value', 'planned_location': {}, 'task_duration': {'seconds': 751, 'nanos': 543}, 'target_time_window': {'start_time': {}, 'end_time': {}}, 'journey_sharing_info': {'remaining_vehicle_journey_segments': [{'stop': {'planned_location': {}, 'tasks': [{'task_id': 'task_id_value', 'task_duration': {}, 'target_time_window': {}}], 'state': 1}, 'driving_distance_meters': {'value': 541}, 'driving_duration': {}, 'path': {}}], 'last_location': {'location': {}, 'horizontal_accuracy': {'value': 0.541}, 'latlng_accuracy': {}, 'heading': {}, 'bearing_accuracy': {}, 'heading_accuracy': {}, 'altitude': {}, 'vertical_accuracy': {}, 'altitude_accuracy': {}, 'speed_kmph': {}, 'speed': {}, 'speed_accuracy': {}, 'update_time': {}, 'server_time': {}, 'location_sensor': 1, 'is_road_snapped': {'value': True}, 'is_gps_sensor_enabled': {}, 'time_since_update': {}, 'num_stale_updates': {}, 'raw_location': {}, 'raw_location_time': {}, 'raw_location_sensor': 1, 'raw_location_accuracy': {}, 'supplemental_location': {}, 'supplemental_location_time': {}, 'supplemental_location_sensor': 1, 'supplemental_location_accuracy': {}, 'road_snapped': True}, 'last_location_snappable': True}, 'task_tracking_view_config': {'route_polyline_points_visibility': {'remaining_stop_count_threshold': 3219, 'duration_until_estimated_arrival_time_threshold': {}, 'remaining_driving_distance_meters_threshold': 4561, 'always': True, 'never': True}, 'estimated_arrival_time_visibility': {}, 'estimated_task_completion_time_visibility': {}, 'remaining_driving_distance_visibility': {}, 'remaining_stop_count_visibility': {}, 'vehicle_location_visibility': {}}, 'attributes': [{'key': 'key_value', 'string_value': 'string_value_value', 'bool_value': True, 'number_value': 0.1285}]}
    test_field = delivery_api.CreateTaskRequest.meta.fields['task']

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
    for (field, value) in request_init['task'].items():
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
                for i in range(0, len(request_init['task'][field])):
                    del request_init['task'][field][i][subfield]
            else:
                del request_init['task'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tasks.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_task(request)
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

def test_create_task_rest_required_fields(request_type=delivery_api.CreateTaskRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['task_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'taskId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'taskId' in jsonified_request
    assert jsonified_request['taskId'] == request_init['task_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['taskId'] = 'task_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_task._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header', 'task_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'taskId' in jsonified_request
    assert jsonified_request['taskId'] == 'task_id_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tasks.Task()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tasks.Task.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_task(request)
            expected_params = [('taskId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_task_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_task._get_unset_required_fields({})
    assert set(unset_fields) == set(('header', 'taskId')) & set(('parent', 'taskId', 'task'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_task_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_create_task') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_create_task') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.CreateTaskRequest.pb(delivery_api.CreateTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tasks.Task.to_json(tasks.Task())
        request = delivery_api.CreateTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tasks.Task()
        client.create_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_task_rest_bad_request(transport: str='rest', request_type=delivery_api.CreateTaskRequest):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_task(request)

def test_create_task_rest_flattened():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tasks.Task()
        sample_request = {'parent': 'providers/sample1'}
        mock_args = dict(parent='parent_value', task=tasks.Task(name='name_value'), task_id='task_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tasks.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=providers/*}/tasks' % client.transport._host, args[1])

def test_create_task_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_task(delivery_api.CreateTaskRequest(), parent='parent_value', task=tasks.Task(name='name_value'), task_id='task_id_value')

def test_create_task_rest_error():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.GetTaskRequest, dict])
def test_get_task_rest(request_type):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'providers/sample1/tasks/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tasks.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_task(request)
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

def test_get_task_rest_required_fields(request_type=delivery_api.GetTaskRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_task._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tasks.Task()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tasks.Task.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_task(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_task_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_task._get_unset_required_fields({})
    assert set(unset_fields) == set(('header',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_task_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_get_task') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_get_task') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.GetTaskRequest.pb(delivery_api.GetTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tasks.Task.to_json(tasks.Task())
        request = delivery_api.GetTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tasks.Task()
        client.get_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_task_rest_bad_request(transport: str='rest', request_type=delivery_api.GetTaskRequest):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'providers/sample1/tasks/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_task(request)

def test_get_task_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tasks.Task()
        sample_request = {'name': 'providers/sample1/tasks/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tasks.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=providers/*/tasks/*}' % client.transport._host, args[1])

def test_get_task_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_task(delivery_api.GetTaskRequest(), name='name_value')

def test_get_task_rest_error():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.SearchTasksRequest, dict])
def test_search_tasks_rest(request_type):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.SearchTasksResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.SearchTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_tasks(request)
    assert isinstance(response, pagers.SearchTasksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_tasks_rest_required_fields(request_type=delivery_api.SearchTasksRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['tracking_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'trackingId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_tasks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'trackingId' in jsonified_request
    assert jsonified_request['trackingId'] == request_init['tracking_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['trackingId'] = 'tracking_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_tasks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header', 'page_size', 'page_token', 'tracking_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'trackingId' in jsonified_request
    assert jsonified_request['trackingId'] == 'tracking_id_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_api.SearchTasksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_api.SearchTasksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_tasks(request)
            expected_params = [('trackingId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_tasks_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_tasks._get_unset_required_fields({})
    assert set(unset_fields) == set(('header', 'pageSize', 'pageToken', 'trackingId')) & set(('parent', 'trackingId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_tasks_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_search_tasks') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_search_tasks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.SearchTasksRequest.pb(delivery_api.SearchTasksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_api.SearchTasksResponse.to_json(delivery_api.SearchTasksResponse())
        request = delivery_api.SearchTasksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_api.SearchTasksResponse()
        client.search_tasks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_tasks_rest_bad_request(transport: str='rest', request_type=delivery_api.SearchTasksRequest):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_tasks(request)

def test_search_tasks_rest_flattened():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.SearchTasksResponse()
        sample_request = {'parent': 'providers/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.SearchTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_tasks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=providers/*}/tasks:search' % client.transport._host, args[1])

def test_search_tasks_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_tasks(delivery_api.SearchTasksRequest(), parent='parent_value')

def test_search_tasks_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.SearchTasksResponse(tasks=[], next_page_token='def'), delivery_api.SearchTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.SearchTasksResponse(tasks=[tasks.Task(), tasks.Task()]))
        response = response + response
        response = tuple((delivery_api.SearchTasksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'providers/sample1'}
        pager = client.search_tasks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tasks.Task) for i in results))
        pages = list(client.search_tasks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [delivery_api.UpdateTaskRequest, dict])
def test_update_task_rest(request_type):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'task': {'name': 'providers/sample1/tasks/sample2'}}
    request_init['task'] = {'name': 'providers/sample1/tasks/sample2', 'type_': 1, 'state': 1, 'task_outcome': 1, 'task_outcome_time': {'seconds': 751, 'nanos': 543}, 'task_outcome_location': {'point': {'latitude': 0.86, 'longitude': 0.971}}, 'task_outcome_location_source': 2, 'tracking_id': 'tracking_id_value', 'delivery_vehicle_id': 'delivery_vehicle_id_value', 'planned_location': {}, 'task_duration': {'seconds': 751, 'nanos': 543}, 'target_time_window': {'start_time': {}, 'end_time': {}}, 'journey_sharing_info': {'remaining_vehicle_journey_segments': [{'stop': {'planned_location': {}, 'tasks': [{'task_id': 'task_id_value', 'task_duration': {}, 'target_time_window': {}}], 'state': 1}, 'driving_distance_meters': {'value': 541}, 'driving_duration': {}, 'path': {}}], 'last_location': {'location': {}, 'horizontal_accuracy': {'value': 0.541}, 'latlng_accuracy': {}, 'heading': {}, 'bearing_accuracy': {}, 'heading_accuracy': {}, 'altitude': {}, 'vertical_accuracy': {}, 'altitude_accuracy': {}, 'speed_kmph': {}, 'speed': {}, 'speed_accuracy': {}, 'update_time': {}, 'server_time': {}, 'location_sensor': 1, 'is_road_snapped': {'value': True}, 'is_gps_sensor_enabled': {}, 'time_since_update': {}, 'num_stale_updates': {}, 'raw_location': {}, 'raw_location_time': {}, 'raw_location_sensor': 1, 'raw_location_accuracy': {}, 'supplemental_location': {}, 'supplemental_location_time': {}, 'supplemental_location_sensor': 1, 'supplemental_location_accuracy': {}, 'road_snapped': True}, 'last_location_snappable': True}, 'task_tracking_view_config': {'route_polyline_points_visibility': {'remaining_stop_count_threshold': 3219, 'duration_until_estimated_arrival_time_threshold': {}, 'remaining_driving_distance_meters_threshold': 4561, 'always': True, 'never': True}, 'estimated_arrival_time_visibility': {}, 'estimated_task_completion_time_visibility': {}, 'remaining_driving_distance_visibility': {}, 'remaining_stop_count_visibility': {}, 'vehicle_location_visibility': {}}, 'attributes': [{'key': 'key_value', 'string_value': 'string_value_value', 'bool_value': True, 'number_value': 0.1285}]}
    test_field = delivery_api.UpdateTaskRequest.meta.fields['task']

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
    for (field, value) in request_init['task'].items():
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
                for i in range(0, len(request_init['task'][field])):
                    del request_init['task'][field][i][subfield]
            else:
                del request_init['task'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tasks.Task(name='name_value', type_=tasks.Task.Type.PICKUP, state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED, task_outcome_location_source=tasks.Task.TaskOutcomeLocationSource.PROVIDER, tracking_id='tracking_id_value', delivery_vehicle_id='delivery_vehicle_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tasks.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_task(request)
    assert isinstance(response, tasks.Task)
    assert response.name == 'name_value'
    assert response.type_ == tasks.Task.Type.PICKUP
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED
    assert response.task_outcome_location_source == tasks.Task.TaskOutcomeLocationSource.PROVIDER
    assert response.tracking_id == 'tracking_id_value'
    assert response.delivery_vehicle_id == 'delivery_vehicle_id_value'

def test_update_task_rest_required_fields(request_type=delivery_api.UpdateTaskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_task._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tasks.Task()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tasks.Task.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_task(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_task_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_task._get_unset_required_fields({})
    assert set(unset_fields) == set(('header', 'updateMask')) & set(('task', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_task_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_update_task') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_update_task') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.UpdateTaskRequest.pb(delivery_api.UpdateTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tasks.Task.to_json(tasks.Task())
        request = delivery_api.UpdateTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tasks.Task()
        client.update_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_task_rest_bad_request(transport: str='rest', request_type=delivery_api.UpdateTaskRequest):
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'task': {'name': 'providers/sample1/tasks/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_task(request)

def test_update_task_rest_flattened():
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tasks.Task()
        sample_request = {'task': {'name': 'providers/sample1/tasks/sample2'}}
        mock_args = dict(task=tasks.Task(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tasks.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{task.name=providers/*/tasks/*}' % client.transport._host, args[1])

def test_update_task_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_task(delivery_api.UpdateTaskRequest(), task=tasks.Task(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_task_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.ListTasksRequest, dict])
def test_list_tasks_rest(request_type):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.ListTasksResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.ListTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tasks(request)
    assert isinstance(response, pagers.ListTasksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_tasks_rest_required_fields(request_type=delivery_api.ListTasksRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tasks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tasks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'header', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_api.ListTasksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_api.ListTasksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tasks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tasks_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tasks._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'header', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tasks_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_list_tasks') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_list_tasks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.ListTasksRequest.pb(delivery_api.ListTasksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_api.ListTasksResponse.to_json(delivery_api.ListTasksResponse())
        request = delivery_api.ListTasksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_api.ListTasksResponse()
        client.list_tasks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tasks_rest_bad_request(transport: str='rest', request_type=delivery_api.ListTasksRequest):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tasks(request)

def test_list_tasks_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.ListTasksResponse()
        sample_request = {'parent': 'providers/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.ListTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tasks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=providers/*}/tasks' % client.transport._host, args[1])

def test_list_tasks_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tasks(delivery_api.ListTasksRequest(), parent='parent_value')

def test_list_tasks_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task(), tasks.Task()], next_page_token='abc'), delivery_api.ListTasksResponse(tasks=[], next_page_token='def'), delivery_api.ListTasksResponse(tasks=[tasks.Task()], next_page_token='ghi'), delivery_api.ListTasksResponse(tasks=[tasks.Task(), tasks.Task()]))
        response = response + response
        response = tuple((delivery_api.ListTasksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'providers/sample1'}
        pager = client.list_tasks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tasks.Task) for i in results))
        pages = list(client.list_tasks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [delivery_api.GetTaskTrackingInfoRequest, dict])
def test_get_task_tracking_info_rest(request_type):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'providers/sample1/taskTrackingInfo/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = task_tracking_info.TaskTrackingInfo(name='name_value', tracking_id='tracking_id_value', state=tasks.Task.State.OPEN, task_outcome=tasks.Task.TaskOutcome.SUCCEEDED)
        response_value = Response()
        response_value.status_code = 200
        return_value = task_tracking_info.TaskTrackingInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_task_tracking_info(request)
    assert isinstance(response, task_tracking_info.TaskTrackingInfo)
    assert response.name == 'name_value'
    assert response.tracking_id == 'tracking_id_value'
    assert response.state == tasks.Task.State.OPEN
    assert response.task_outcome == tasks.Task.TaskOutcome.SUCCEEDED

def test_get_task_tracking_info_rest_required_fields(request_type=delivery_api.GetTaskTrackingInfoRequest):
    if False:
        return 10
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_task_tracking_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_task_tracking_info._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('header',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = task_tracking_info.TaskTrackingInfo()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = task_tracking_info.TaskTrackingInfo.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_task_tracking_info(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_task_tracking_info_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_task_tracking_info._get_unset_required_fields({})
    assert set(unset_fields) == set(('header',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_task_tracking_info_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_get_task_tracking_info') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_get_task_tracking_info') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.GetTaskTrackingInfoRequest.pb(delivery_api.GetTaskTrackingInfoRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = task_tracking_info.TaskTrackingInfo.to_json(task_tracking_info.TaskTrackingInfo())
        request = delivery_api.GetTaskTrackingInfoRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = task_tracking_info.TaskTrackingInfo()
        client.get_task_tracking_info(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_task_tracking_info_rest_bad_request(transport: str='rest', request_type=delivery_api.GetTaskTrackingInfoRequest):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'providers/sample1/taskTrackingInfo/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_task_tracking_info(request)

def test_get_task_tracking_info_rest_flattened():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = task_tracking_info.TaskTrackingInfo()
        sample_request = {'name': 'providers/sample1/taskTrackingInfo/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = task_tracking_info.TaskTrackingInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_task_tracking_info(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=providers/*/taskTrackingInfo/*}' % client.transport._host, args[1])

def test_get_task_tracking_info_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_task_tracking_info(delivery_api.GetTaskTrackingInfoRequest(), name='name_value')

def test_get_task_tracking_info_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [delivery_api.ListDeliveryVehiclesRequest, dict])
def test_list_delivery_vehicles_rest(request_type):
    if False:
        while True:
            i = 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.ListDeliveryVehiclesResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.ListDeliveryVehiclesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_delivery_vehicles(request)
    assert isinstance(response, pagers.ListDeliveryVehiclesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_delivery_vehicles_rest_required_fields(request_type=delivery_api.ListDeliveryVehiclesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DeliveryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_delivery_vehicles._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_delivery_vehicles._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'header', 'page_size', 'page_token', 'viewport'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = delivery_api.ListDeliveryVehiclesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = delivery_api.ListDeliveryVehiclesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_delivery_vehicles(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_delivery_vehicles_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_delivery_vehicles._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'header', 'pageSize', 'pageToken', 'viewport')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_delivery_vehicles_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DeliveryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DeliveryServiceRestInterceptor())
    client = DeliveryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'post_list_delivery_vehicles') as post, mock.patch.object(transports.DeliveryServiceRestInterceptor, 'pre_list_delivery_vehicles') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = delivery_api.ListDeliveryVehiclesRequest.pb(delivery_api.ListDeliveryVehiclesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = delivery_api.ListDeliveryVehiclesResponse.to_json(delivery_api.ListDeliveryVehiclesResponse())
        request = delivery_api.ListDeliveryVehiclesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = delivery_api.ListDeliveryVehiclesResponse()
        client.list_delivery_vehicles(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_delivery_vehicles_rest_bad_request(transport: str='rest', request_type=delivery_api.ListDeliveryVehiclesRequest):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'providers/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_delivery_vehicles(request)

def test_list_delivery_vehicles_rest_flattened():
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = delivery_api.ListDeliveryVehiclesResponse()
        sample_request = {'parent': 'providers/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = delivery_api.ListDeliveryVehiclesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_delivery_vehicles(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=providers/*}/deliveryVehicles' % client.transport._host, args[1])

def test_list_delivery_vehicles_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_delivery_vehicles(delivery_api.ListDeliveryVehiclesRequest(), parent='parent_value')

def test_list_delivery_vehicles_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()], next_page_token='abc'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[], next_page_token='def'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle()], next_page_token='ghi'), delivery_api.ListDeliveryVehiclesResponse(delivery_vehicles=[delivery_vehicles.DeliveryVehicle(), delivery_vehicles.DeliveryVehicle()]))
        response = response + response
        response = tuple((delivery_api.ListDeliveryVehiclesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'providers/sample1'}
        pager = client.list_delivery_vehicles(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, delivery_vehicles.DeliveryVehicle) for i in results))
        pages = list(client.list_delivery_vehicles(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DeliveryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DeliveryServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DeliveryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DeliveryServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DeliveryServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DeliveryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DeliveryServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.DeliveryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DeliveryServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.DeliveryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DeliveryServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceGrpcAsyncIOTransport, transports.DeliveryServiceRestTransport])
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
        print('Hello World!')
    transport = DeliveryServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DeliveryServiceGrpcTransport)

def test_delivery_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DeliveryServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_delivery_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.maps.fleetengine_delivery_v1.services.delivery_service.transports.DeliveryServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DeliveryServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_delivery_vehicle', 'get_delivery_vehicle', 'update_delivery_vehicle', 'batch_create_tasks', 'create_task', 'get_task', 'search_tasks', 'update_task', 'list_tasks', 'get_task_tracking_info', 'list_delivery_vehicles')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_delivery_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.maps.fleetengine_delivery_v1.services.delivery_service.transports.DeliveryServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DeliveryServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_delivery_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.maps.fleetengine_delivery_v1.services.delivery_service.transports.DeliveryServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DeliveryServiceTransport()
        adc.assert_called_once()

def test_delivery_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DeliveryServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceGrpcAsyncIOTransport])
def test_delivery_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceGrpcAsyncIOTransport, transports.DeliveryServiceRestTransport])
def test_delivery_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DeliveryServiceGrpcTransport, grpc_helpers), (transports.DeliveryServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_delivery_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('fleetengine.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='fleetengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceGrpcAsyncIOTransport])
def test_delivery_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_delivery_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.DeliveryServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_delivery_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='fleetengine.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('fleetengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://fleetengine.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_delivery_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='fleetengine.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('fleetengine.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://fleetengine.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_delivery_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = DeliveryServiceClient(credentials=creds1, transport=transport_name)
    client2 = DeliveryServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_delivery_vehicle._session
    session2 = client2.transport.create_delivery_vehicle._session
    assert session1 != session2
    session1 = client1.transport.get_delivery_vehicle._session
    session2 = client2.transport.get_delivery_vehicle._session
    assert session1 != session2
    session1 = client1.transport.update_delivery_vehicle._session
    session2 = client2.transport.update_delivery_vehicle._session
    assert session1 != session2
    session1 = client1.transport.batch_create_tasks._session
    session2 = client2.transport.batch_create_tasks._session
    assert session1 != session2
    session1 = client1.transport.create_task._session
    session2 = client2.transport.create_task._session
    assert session1 != session2
    session1 = client1.transport.get_task._session
    session2 = client2.transport.get_task._session
    assert session1 != session2
    session1 = client1.transport.search_tasks._session
    session2 = client2.transport.search_tasks._session
    assert session1 != session2
    session1 = client1.transport.update_task._session
    session2 = client2.transport.update_task._session
    assert session1 != session2
    session1 = client1.transport.list_tasks._session
    session2 = client2.transport.list_tasks._session
    assert session1 != session2
    session1 = client1.transport.get_task_tracking_info._session
    session2 = client2.transport.get_task_tracking_info._session
    assert session1 != session2
    session1 = client1.transport.list_delivery_vehicles._session
    session2 = client2.transport.list_delivery_vehicles._session
    assert session1 != session2

def test_delivery_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DeliveryServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_delivery_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DeliveryServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceGrpcAsyncIOTransport])
def test_delivery_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.DeliveryServiceGrpcTransport, transports.DeliveryServiceGrpcAsyncIOTransport])
def test_delivery_service_transport_channel_mtls_with_adc(transport_class):
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

def test_delivery_vehicle_path():
    if False:
        print('Hello World!')
    provider = 'squid'
    vehicle = 'clam'
    expected = 'providers/{provider}/deliveryVehicles/{vehicle}'.format(provider=provider, vehicle=vehicle)
    actual = DeliveryServiceClient.delivery_vehicle_path(provider, vehicle)
    assert expected == actual

def test_parse_delivery_vehicle_path():
    if False:
        while True:
            i = 10
    expected = {'provider': 'whelk', 'vehicle': 'octopus'}
    path = DeliveryServiceClient.delivery_vehicle_path(**expected)
    actual = DeliveryServiceClient.parse_delivery_vehicle_path(path)
    assert expected == actual

def test_task_path():
    if False:
        for i in range(10):
            print('nop')
    provider = 'oyster'
    task = 'nudibranch'
    expected = 'providers/{provider}/tasks/{task}'.format(provider=provider, task=task)
    actual = DeliveryServiceClient.task_path(provider, task)
    assert expected == actual

def test_parse_task_path():
    if False:
        i = 10
        return i + 15
    expected = {'provider': 'cuttlefish', 'task': 'mussel'}
    path = DeliveryServiceClient.task_path(**expected)
    actual = DeliveryServiceClient.parse_task_path(path)
    assert expected == actual

def test_task_tracking_info_path():
    if False:
        return 10
    provider = 'winkle'
    tracking = 'nautilus'
    expected = 'providers/{provider}/taskTrackingInfo/{tracking}'.format(provider=provider, tracking=tracking)
    actual = DeliveryServiceClient.task_tracking_info_path(provider, tracking)
    assert expected == actual

def test_parse_task_tracking_info_path():
    if False:
        print('Hello World!')
    expected = {'provider': 'scallop', 'tracking': 'abalone'}
    path = DeliveryServiceClient.task_tracking_info_path(**expected)
    actual = DeliveryServiceClient.parse_task_tracking_info_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DeliveryServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'clam'}
    path = DeliveryServiceClient.common_billing_account_path(**expected)
    actual = DeliveryServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DeliveryServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'octopus'}
    path = DeliveryServiceClient.common_folder_path(**expected)
    actual = DeliveryServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DeliveryServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = DeliveryServiceClient.common_organization_path(**expected)
    actual = DeliveryServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = DeliveryServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'mussel'}
    path = DeliveryServiceClient.common_project_path(**expected)
    actual = DeliveryServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DeliveryServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = DeliveryServiceClient.common_location_path(**expected)
    actual = DeliveryServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DeliveryServiceTransport, '_prep_wrapped_messages') as prep:
        client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DeliveryServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DeliveryServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DeliveryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = DeliveryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DeliveryServiceClient, transports.DeliveryServiceGrpcTransport), (DeliveryServiceAsyncClient, transports.DeliveryServiceGrpcAsyncIOTransport)])
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