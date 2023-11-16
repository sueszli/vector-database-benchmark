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
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.maps.mapsplatformdatasets_v1alpha.services.maps_platform_datasets_v1_alpha import MapsPlatformDatasetsV1AlphaAsyncClient, MapsPlatformDatasetsV1AlphaClient, pagers, transports
from google.maps.mapsplatformdatasets_v1alpha.types import dataset as gmm_dataset
from google.maps.mapsplatformdatasets_v1alpha.types import maps_platform_datasets
from google.maps.mapsplatformdatasets_v1alpha.types import data_source
from google.maps.mapsplatformdatasets_v1alpha.types import dataset

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert MapsPlatformDatasetsV1AlphaClient._get_default_mtls_endpoint(None) is None
    assert MapsPlatformDatasetsV1AlphaClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert MapsPlatformDatasetsV1AlphaClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert MapsPlatformDatasetsV1AlphaClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert MapsPlatformDatasetsV1AlphaClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert MapsPlatformDatasetsV1AlphaClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(MapsPlatformDatasetsV1AlphaClient, 'grpc'), (MapsPlatformDatasetsV1AlphaAsyncClient, 'grpc_asyncio'), (MapsPlatformDatasetsV1AlphaClient, 'rest')])
def test_maps_platform_datasets_v1_alpha_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('mapsplatformdatasets.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://mapsplatformdatasets.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc'), (transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.MapsPlatformDatasetsV1AlphaRestTransport, 'rest')])
def test_maps_platform_datasets_v1_alpha_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(MapsPlatformDatasetsV1AlphaClient, 'grpc'), (MapsPlatformDatasetsV1AlphaAsyncClient, 'grpc_asyncio'), (MapsPlatformDatasetsV1AlphaClient, 'rest')])
def test_maps_platform_datasets_v1_alpha_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('mapsplatformdatasets.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://mapsplatformdatasets.googleapis.com')

def test_maps_platform_datasets_v1_alpha_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = MapsPlatformDatasetsV1AlphaClient.get_transport_class()
    available_transports = [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaRestTransport]
    assert transport in available_transports
    transport = MapsPlatformDatasetsV1AlphaClient.get_transport_class('grpc')
    assert transport == transports.MapsPlatformDatasetsV1AlphaGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc'), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio'), (MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaRestTransport, 'rest')])
@mock.patch.object(MapsPlatformDatasetsV1AlphaClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MapsPlatformDatasetsV1AlphaClient))
@mock.patch.object(MapsPlatformDatasetsV1AlphaAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MapsPlatformDatasetsV1AlphaAsyncClient))
def test_maps_platform_datasets_v1_alpha_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(MapsPlatformDatasetsV1AlphaClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(MapsPlatformDatasetsV1AlphaClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc', 'true'), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc', 'false'), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaRestTransport, 'rest', 'true'), (MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaRestTransport, 'rest', 'false')])
@mock.patch.object(MapsPlatformDatasetsV1AlphaClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MapsPlatformDatasetsV1AlphaClient))
@mock.patch.object(MapsPlatformDatasetsV1AlphaAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MapsPlatformDatasetsV1AlphaAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_maps_platform_datasets_v1_alpha_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [MapsPlatformDatasetsV1AlphaClient, MapsPlatformDatasetsV1AlphaAsyncClient])
@mock.patch.object(MapsPlatformDatasetsV1AlphaClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MapsPlatformDatasetsV1AlphaClient))
@mock.patch.object(MapsPlatformDatasetsV1AlphaAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MapsPlatformDatasetsV1AlphaAsyncClient))
def test_maps_platform_datasets_v1_alpha_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc'), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio'), (MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaRestTransport, 'rest')])
def test_maps_platform_datasets_v1_alpha_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc', grpc_helpers), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaRestTransport, 'rest', None)])
def test_maps_platform_datasets_v1_alpha_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_maps_platform_datasets_v1_alpha_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.maps.mapsplatformdatasets_v1alpha.services.maps_platform_datasets_v1_alpha.transports.MapsPlatformDatasetsV1AlphaGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = MapsPlatformDatasetsV1AlphaClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport, 'grpc', grpc_helpers), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_maps_platform_datasets_v1_alpha_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('mapsplatformdatasets.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='mapsplatformdatasets.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [maps_platform_datasets.CreateDatasetRequest, dict])
def test_create_dataset(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        call.return_value = gmm_dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=gmm_dataset.State.STATE_IMPORTING, version_description='version_description_value')
        response = client.create_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.CreateDatasetRequest()
    assert isinstance(response, gmm_dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == gmm_dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

def test_create_dataset_empty_call():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        client.create_dataset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.CreateDatasetRequest()

@pytest.mark.asyncio
async def test_create_dataset_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.CreateDatasetRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gmm_dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=gmm_dataset.State.STATE_IMPORTING, version_description='version_description_value'))
        response = await client.create_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.CreateDatasetRequest()
    assert isinstance(response, gmm_dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == gmm_dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

@pytest.mark.asyncio
async def test_create_dataset_async_from_dict():
    await test_create_dataset_async(request_type=dict)

def test_create_dataset_field_headers():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.CreateDatasetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        call.return_value = gmm_dataset.Dataset()
        client.create_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_dataset_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.CreateDatasetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gmm_dataset.Dataset())
        await client.create_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_dataset_flattened():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        call.return_value = gmm_dataset.Dataset()
        client.create_dataset(parent='parent_value', dataset=gmm_dataset.Dataset(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].dataset
        mock_val = gmm_dataset.Dataset(name='name_value')
        assert arg == mock_val

def test_create_dataset_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_dataset(maps_platform_datasets.CreateDatasetRequest(), parent='parent_value', dataset=gmm_dataset.Dataset(name='name_value'))

@pytest.mark.asyncio
async def test_create_dataset_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_dataset), '__call__') as call:
        call.return_value = gmm_dataset.Dataset()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gmm_dataset.Dataset())
        response = await client.create_dataset(parent='parent_value', dataset=gmm_dataset.Dataset(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].dataset
        mock_val = gmm_dataset.Dataset(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_dataset_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_dataset(maps_platform_datasets.CreateDatasetRequest(), parent='parent_value', dataset=gmm_dataset.Dataset(name='name_value'))

@pytest.mark.parametrize('request_type', [maps_platform_datasets.UpdateDatasetMetadataRequest, dict])
def test_update_dataset_metadata(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        call.return_value = gmm_dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=gmm_dataset.State.STATE_IMPORTING, version_description='version_description_value')
        response = client.update_dataset_metadata(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.UpdateDatasetMetadataRequest()
    assert isinstance(response, gmm_dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == gmm_dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

def test_update_dataset_metadata_empty_call():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        client.update_dataset_metadata()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.UpdateDatasetMetadataRequest()

@pytest.mark.asyncio
async def test_update_dataset_metadata_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.UpdateDatasetMetadataRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gmm_dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=gmm_dataset.State.STATE_IMPORTING, version_description='version_description_value'))
        response = await client.update_dataset_metadata(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.UpdateDatasetMetadataRequest()
    assert isinstance(response, gmm_dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == gmm_dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

@pytest.mark.asyncio
async def test_update_dataset_metadata_async_from_dict():
    await test_update_dataset_metadata_async(request_type=dict)

def test_update_dataset_metadata_field_headers():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.UpdateDatasetMetadataRequest()
    request.dataset.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        call.return_value = gmm_dataset.Dataset()
        client.update_dataset_metadata(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_dataset_metadata_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.UpdateDatasetMetadataRequest()
    request.dataset.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gmm_dataset.Dataset())
        await client.update_dataset_metadata(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dataset.name=name_value') in kw['metadata']

def test_update_dataset_metadata_flattened():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        call.return_value = gmm_dataset.Dataset()
        client.update_dataset_metadata(dataset=gmm_dataset.Dataset(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = gmm_dataset.Dataset(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_dataset_metadata_flattened_error():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_dataset_metadata(maps_platform_datasets.UpdateDatasetMetadataRequest(), dataset=gmm_dataset.Dataset(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_dataset_metadata_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dataset_metadata), '__call__') as call:
        call.return_value = gmm_dataset.Dataset()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gmm_dataset.Dataset())
        response = await client.update_dataset_metadata(dataset=gmm_dataset.Dataset(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dataset
        mock_val = gmm_dataset.Dataset(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_dataset_metadata_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_dataset_metadata(maps_platform_datasets.UpdateDatasetMetadataRequest(), dataset=gmm_dataset.Dataset(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [maps_platform_datasets.GetDatasetRequest, dict])
def test_get_dataset(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        call.return_value = dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=dataset.State.STATE_IMPORTING, version_description='version_description_value')
        response = client.get_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.GetDatasetRequest()
    assert isinstance(response, dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

def test_get_dataset_empty_call():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        client.get_dataset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.GetDatasetRequest()

@pytest.mark.asyncio
async def test_get_dataset_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.GetDatasetRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=dataset.State.STATE_IMPORTING, version_description='version_description_value'))
        response = await client.get_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.GetDatasetRequest()
    assert isinstance(response, dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

@pytest.mark.asyncio
async def test_get_dataset_async_from_dict():
    await test_get_dataset_async(request_type=dict)

def test_get_dataset_field_headers():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.GetDatasetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        call.return_value = dataset.Dataset()
        client.get_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_dataset_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.GetDatasetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.Dataset())
        await client.get_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_dataset_flattened():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        call.return_value = dataset.Dataset()
        client.get_dataset(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_dataset_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_dataset(maps_platform_datasets.GetDatasetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_dataset_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dataset), '__call__') as call:
        call.return_value = dataset.Dataset()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dataset.Dataset())
        response = await client.get_dataset(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_dataset_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_dataset(maps_platform_datasets.GetDatasetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.ListDatasetVersionsRequest, dict])
def test_list_dataset_versions(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetVersionsResponse(next_page_token='next_page_token_value')
        response = client.list_dataset_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.ListDatasetVersionsRequest()
    assert isinstance(response, pagers.ListDatasetVersionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_dataset_versions_empty_call():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        client.list_dataset_versions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.ListDatasetVersionsRequest()

@pytest.mark.asyncio
async def test_list_dataset_versions_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.ListDatasetVersionsRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(maps_platform_datasets.ListDatasetVersionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_dataset_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.ListDatasetVersionsRequest()
    assert isinstance(response, pagers.ListDatasetVersionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_dataset_versions_async_from_dict():
    await test_list_dataset_versions_async(request_type=dict)

def test_list_dataset_versions_field_headers():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.ListDatasetVersionsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetVersionsResponse()
        client.list_dataset_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_dataset_versions_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.ListDatasetVersionsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(maps_platform_datasets.ListDatasetVersionsResponse())
        await client.list_dataset_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_list_dataset_versions_flattened():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetVersionsResponse()
        client.list_dataset_versions(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_list_dataset_versions_flattened_error():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_dataset_versions(maps_platform_datasets.ListDatasetVersionsRequest(), name='name_value')

@pytest.mark.asyncio
async def test_list_dataset_versions_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetVersionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(maps_platform_datasets.ListDatasetVersionsResponse())
        response = await client.list_dataset_versions(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_dataset_versions_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_dataset_versions(maps_platform_datasets.ListDatasetVersionsRequest(), name='name_value')

def test_list_dataset_versions_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.side_effect = (maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', ''),)),)
        pager = client.list_dataset_versions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dataset.Dataset) for i in results))

def test_list_dataset_versions_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__') as call:
        call.side_effect = (maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        pages = list(client.list_dataset_versions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_dataset_versions_async_pager():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        async_pager = await client.list_dataset_versions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dataset.Dataset) for i in responses))

@pytest.mark.asyncio
async def test_list_dataset_versions_async_pages():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dataset_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_dataset_versions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [maps_platform_datasets.ListDatasetsRequest, dict])
def test_list_datasets(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetsResponse(next_page_token='next_page_token_value')
        response = client.list_datasets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.ListDatasetsRequest()
    assert isinstance(response, pagers.ListDatasetsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_datasets_empty_call():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        client.list_datasets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.ListDatasetsRequest()

@pytest.mark.asyncio
async def test_list_datasets_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.ListDatasetsRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(maps_platform_datasets.ListDatasetsResponse(next_page_token='next_page_token_value'))
        response = await client.list_datasets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.ListDatasetsRequest()
    assert isinstance(response, pagers.ListDatasetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_datasets_async_from_dict():
    await test_list_datasets_async(request_type=dict)

def test_list_datasets_field_headers():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.ListDatasetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetsResponse()
        client.list_datasets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_datasets_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.ListDatasetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(maps_platform_datasets.ListDatasetsResponse())
        await client.list_datasets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_datasets_flattened():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetsResponse()
        client.list_datasets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_datasets_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_datasets(maps_platform_datasets.ListDatasetsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_datasets_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.return_value = maps_platform_datasets.ListDatasetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(maps_platform_datasets.ListDatasetsResponse())
        response = await client.list_datasets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_datasets_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_datasets(maps_platform_datasets.ListDatasetsRequest(), parent='parent_value')

def test_list_datasets_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.side_effect = (maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_datasets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dataset.Dataset) for i in results))

def test_list_datasets_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_datasets), '__call__') as call:
        call.side_effect = (maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        pages = list(client.list_datasets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_datasets_async_pager():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_datasets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        async_pager = await client.list_datasets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dataset.Dataset) for i in responses))

@pytest.mark.asyncio
async def test_list_datasets_async_pages():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_datasets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_datasets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [maps_platform_datasets.DeleteDatasetRequest, dict])
def test_delete_dataset(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        call.return_value = None
        response = client.delete_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.DeleteDatasetRequest()
    assert response is None

def test_delete_dataset_empty_call():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        client.delete_dataset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.DeleteDatasetRequest()

@pytest.mark.asyncio
async def test_delete_dataset_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.DeleteDatasetRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.DeleteDatasetRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_dataset_async_from_dict():
    await test_delete_dataset_async(request_type=dict)

def test_delete_dataset_field_headers():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.DeleteDatasetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        call.return_value = None
        client.delete_dataset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_dataset_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.DeleteDatasetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_dataset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_dataset_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        call.return_value = None
        client.delete_dataset(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_dataset_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_dataset(maps_platform_datasets.DeleteDatasetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_dataset_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dataset), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dataset(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_dataset_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_dataset(maps_platform_datasets.DeleteDatasetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.DeleteDatasetVersionRequest, dict])
def test_delete_dataset_version(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        call.return_value = None
        response = client.delete_dataset_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.DeleteDatasetVersionRequest()
    assert response is None

def test_delete_dataset_version_empty_call():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        client.delete_dataset_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.DeleteDatasetVersionRequest()

@pytest.mark.asyncio
async def test_delete_dataset_version_async(transport: str='grpc_asyncio', request_type=maps_platform_datasets.DeleteDatasetVersionRequest):
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dataset_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == maps_platform_datasets.DeleteDatasetVersionRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_dataset_version_async_from_dict():
    await test_delete_dataset_version_async(request_type=dict)

def test_delete_dataset_version_field_headers():
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.DeleteDatasetVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        call.return_value = None
        client.delete_dataset_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_dataset_version_field_headers_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = maps_platform_datasets.DeleteDatasetVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_dataset_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_dataset_version_flattened():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        call.return_value = None
        client.delete_dataset_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_dataset_version_flattened_error():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_dataset_version(maps_platform_datasets.DeleteDatasetVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_dataset_version_flattened_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dataset_version), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dataset_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_dataset_version_flattened_error_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_dataset_version(maps_platform_datasets.DeleteDatasetVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.CreateDatasetRequest, dict])
def test_create_dataset_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['dataset'] = {'name': 'name_value', 'display_name': 'display_name_value', 'description': 'description_value', 'version_id': 'version_id_value', 'usage': [1], 'local_file_source': {'filename': 'filename_value', 'file_format': 1}, 'gcs_source': {'input_uri': 'input_uri_value', 'file_format': 1}, 'status': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'version_create_time': {}, 'version_description': 'version_description_value'}
    test_field = maps_platform_datasets.CreateDatasetRequest.meta.fields['dataset']

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
        return_value = gmm_dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=gmm_dataset.State.STATE_IMPORTING, version_description='version_description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gmm_dataset.Dataset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_dataset(request)
    assert isinstance(response, gmm_dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == gmm_dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

def test_create_dataset_rest_required_fields(request_type=maps_platform_datasets.CreateDatasetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dataset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dataset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gmm_dataset.Dataset()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gmm_dataset.Dataset.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_dataset(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_dataset_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_dataset._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'dataset'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_dataset_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'post_create_dataset') as post, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_create_dataset') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = maps_platform_datasets.CreateDatasetRequest.pb(maps_platform_datasets.CreateDatasetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gmm_dataset.Dataset.to_json(gmm_dataset.Dataset())
        request = maps_platform_datasets.CreateDatasetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gmm_dataset.Dataset()
        client.create_dataset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_dataset_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.CreateDatasetRequest):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_dataset(request)

def test_create_dataset_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gmm_dataset.Dataset()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', dataset=gmm_dataset.Dataset(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gmm_dataset.Dataset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_dataset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*}/datasets' % client.transport._host, args[1])

def test_create_dataset_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_dataset(maps_platform_datasets.CreateDatasetRequest(), parent='parent_value', dataset=gmm_dataset.Dataset(name='name_value'))

def test_create_dataset_rest_error():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.UpdateDatasetMetadataRequest, dict])
def test_update_dataset_metadata_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dataset': {'name': 'projects/sample1/datasets/sample2'}}
    request_init['dataset'] = {'name': 'projects/sample1/datasets/sample2', 'display_name': 'display_name_value', 'description': 'description_value', 'version_id': 'version_id_value', 'usage': [1], 'local_file_source': {'filename': 'filename_value', 'file_format': 1}, 'gcs_source': {'input_uri': 'input_uri_value', 'file_format': 1}, 'status': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'version_create_time': {}, 'version_description': 'version_description_value'}
    test_field = maps_platform_datasets.UpdateDatasetMetadataRequest.meta.fields['dataset']

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
        return_value = gmm_dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=gmm_dataset.State.STATE_IMPORTING, version_description='version_description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gmm_dataset.Dataset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_dataset_metadata(request)
    assert isinstance(response, gmm_dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [gmm_dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == gmm_dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

def test_update_dataset_metadata_rest_required_fields(request_type=maps_platform_datasets.UpdateDatasetMetadataRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dataset_metadata._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dataset_metadata._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gmm_dataset.Dataset()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gmm_dataset.Dataset.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_dataset_metadata(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_dataset_metadata_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_dataset_metadata._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('dataset',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_dataset_metadata_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'post_update_dataset_metadata') as post, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_update_dataset_metadata') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = maps_platform_datasets.UpdateDatasetMetadataRequest.pb(maps_platform_datasets.UpdateDatasetMetadataRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gmm_dataset.Dataset.to_json(gmm_dataset.Dataset())
        request = maps_platform_datasets.UpdateDatasetMetadataRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gmm_dataset.Dataset()
        client.update_dataset_metadata(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_dataset_metadata_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.UpdateDatasetMetadataRequest):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dataset': {'name': 'projects/sample1/datasets/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_dataset_metadata(request)

def test_update_dataset_metadata_rest_flattened():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gmm_dataset.Dataset()
        sample_request = {'dataset': {'name': 'projects/sample1/datasets/sample2'}}
        mock_args = dict(dataset=gmm_dataset.Dataset(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gmm_dataset.Dataset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_dataset_metadata(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{dataset.name=projects/*/datasets/*}' % client.transport._host, args[1])

def test_update_dataset_metadata_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_dataset_metadata(maps_platform_datasets.UpdateDatasetMetadataRequest(), dataset=gmm_dataset.Dataset(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_dataset_metadata_rest_error():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.GetDatasetRequest, dict])
def test_get_dataset_rest(request_type):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dataset.Dataset(name='name_value', display_name='display_name_value', description='description_value', version_id='version_id_value', usage=[dataset.Usage.USAGE_DATA_DRIVEN_STYLING], status=dataset.State.STATE_IMPORTING, version_description='version_description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dataset.Dataset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_dataset(request)
    assert isinstance(response, dataset.Dataset)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.version_id == 'version_id_value'
    assert response.usage == [dataset.Usage.USAGE_DATA_DRIVEN_STYLING]
    assert response.status == dataset.State.STATE_IMPORTING
    assert response.version_description == 'version_description_value'

def test_get_dataset_rest_required_fields(request_type=maps_platform_datasets.GetDatasetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dataset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dataset._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('published_usage',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dataset.Dataset()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dataset.Dataset.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_dataset(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_dataset_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_dataset._get_unset_required_fields({})
    assert set(unset_fields) == set(('publishedUsage',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_dataset_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'post_get_dataset') as post, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_get_dataset') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = maps_platform_datasets.GetDatasetRequest.pb(maps_platform_datasets.GetDatasetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dataset.Dataset.to_json(dataset.Dataset())
        request = maps_platform_datasets.GetDatasetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dataset.Dataset()
        client.get_dataset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_dataset_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.GetDatasetRequest):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_dataset(request)

def test_get_dataset_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dataset.Dataset()
        sample_request = {'name': 'projects/sample1/datasets/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dataset.Dataset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_dataset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/datasets/*}' % client.transport._host, args[1])

def test_get_dataset_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_dataset(maps_platform_datasets.GetDatasetRequest(), name='name_value')

def test_get_dataset_rest_error():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.ListDatasetVersionsRequest, dict])
def test_list_dataset_versions_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = maps_platform_datasets.ListDatasetVersionsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = maps_platform_datasets.ListDatasetVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_dataset_versions(request)
    assert isinstance(response, pagers.ListDatasetVersionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_dataset_versions_rest_required_fields(request_type=maps_platform_datasets.ListDatasetVersionsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dataset_versions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dataset_versions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = maps_platform_datasets.ListDatasetVersionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = maps_platform_datasets.ListDatasetVersionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_dataset_versions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_dataset_versions_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_dataset_versions._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_dataset_versions_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'post_list_dataset_versions') as post, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_list_dataset_versions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = maps_platform_datasets.ListDatasetVersionsRequest.pb(maps_platform_datasets.ListDatasetVersionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = maps_platform_datasets.ListDatasetVersionsResponse.to_json(maps_platform_datasets.ListDatasetVersionsResponse())
        request = maps_platform_datasets.ListDatasetVersionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = maps_platform_datasets.ListDatasetVersionsResponse()
        client.list_dataset_versions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_dataset_versions_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.ListDatasetVersionsRequest):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_dataset_versions(request)

def test_list_dataset_versions_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = maps_platform_datasets.ListDatasetVersionsResponse()
        sample_request = {'name': 'projects/sample1/datasets/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = maps_platform_datasets.ListDatasetVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_dataset_versions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/datasets/*}:listVersions' % client.transport._host, args[1])

def test_list_dataset_versions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_dataset_versions(maps_platform_datasets.ListDatasetVersionsRequest(), name='name_value')

def test_list_dataset_versions_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetVersionsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]))
        response = response + response
        response = tuple((maps_platform_datasets.ListDatasetVersionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'name': 'projects/sample1/datasets/sample2'}
        pager = client.list_dataset_versions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dataset.Dataset) for i in results))
        pages = list(client.list_dataset_versions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [maps_platform_datasets.ListDatasetsRequest, dict])
def test_list_datasets_rest(request_type):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = maps_platform_datasets.ListDatasetsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = maps_platform_datasets.ListDatasetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_datasets(request)
    assert isinstance(response, pagers.ListDatasetsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_datasets_rest_required_fields(request_type=maps_platform_datasets.ListDatasetsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_datasets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_datasets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = maps_platform_datasets.ListDatasetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = maps_platform_datasets.ListDatasetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_datasets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_datasets_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_datasets._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_datasets_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'post_list_datasets') as post, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_list_datasets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = maps_platform_datasets.ListDatasetsRequest.pb(maps_platform_datasets.ListDatasetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = maps_platform_datasets.ListDatasetsResponse.to_json(maps_platform_datasets.ListDatasetsResponse())
        request = maps_platform_datasets.ListDatasetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = maps_platform_datasets.ListDatasetsResponse()
        client.list_datasets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_datasets_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.ListDatasetsRequest):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_datasets(request)

def test_list_datasets_rest_flattened():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = maps_platform_datasets.ListDatasetsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = maps_platform_datasets.ListDatasetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_datasets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*}/datasets' % client.transport._host, args[1])

def test_list_datasets_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_datasets(maps_platform_datasets.ListDatasetsRequest(), parent='parent_value')

def test_list_datasets_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset(), dataset.Dataset()], next_page_token='abc'), maps_platform_datasets.ListDatasetsResponse(datasets=[], next_page_token='def'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset()], next_page_token='ghi'), maps_platform_datasets.ListDatasetsResponse(datasets=[dataset.Dataset(), dataset.Dataset()]))
        response = response + response
        response = tuple((maps_platform_datasets.ListDatasetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_datasets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dataset.Dataset) for i in results))
        pages = list(client.list_datasets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [maps_platform_datasets.DeleteDatasetRequest, dict])
def test_delete_dataset_rest(request_type):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_dataset(request)
    assert response is None

def test_delete_dataset_rest_required_fields(request_type=maps_platform_datasets.DeleteDatasetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dataset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dataset._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_dataset(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_dataset_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_dataset._get_unset_required_fields({})
    assert set(unset_fields) == set(('force',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_dataset_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_delete_dataset') as pre:
        pre.assert_not_called()
        pb_message = maps_platform_datasets.DeleteDatasetRequest.pb(maps_platform_datasets.DeleteDatasetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = maps_platform_datasets.DeleteDatasetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_dataset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_dataset_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.DeleteDatasetRequest):
    if False:
        while True:
            i = 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_dataset(request)

def test_delete_dataset_rest_flattened():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/datasets/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_dataset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/datasets/*}' % client.transport._host, args[1])

def test_delete_dataset_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_dataset(maps_platform_datasets.DeleteDatasetRequest(), name='name_value')

def test_delete_dataset_rest_error():
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [maps_platform_datasets.DeleteDatasetVersionRequest, dict])
def test_delete_dataset_version_rest(request_type):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_dataset_version(request)
    assert response is None

def test_delete_dataset_version_rest_required_fields(request_type=maps_platform_datasets.DeleteDatasetVersionRequest):
    if False:
        return 10
    transport_class = transports.MapsPlatformDatasetsV1AlphaRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dataset_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dataset_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_dataset_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_dataset_version_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_dataset_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_dataset_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MapsPlatformDatasetsV1AlphaRestInterceptor())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MapsPlatformDatasetsV1AlphaRestInterceptor, 'pre_delete_dataset_version') as pre:
        pre.assert_not_called()
        pb_message = maps_platform_datasets.DeleteDatasetVersionRequest.pb(maps_platform_datasets.DeleteDatasetVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = maps_platform_datasets.DeleteDatasetVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_dataset_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_dataset_version_rest_bad_request(transport: str='rest', request_type=maps_platform_datasets.DeleteDatasetVersionRequest):
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/datasets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_dataset_version(request)

def test_delete_dataset_version_rest_flattened():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/datasets/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_dataset_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/datasets/*}:deleteVersion' % client.transport._host, args[1])

def test_delete_dataset_version_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_dataset_version(maps_platform_datasets.DeleteDatasetVersionRequest(), name='name_value')

def test_delete_dataset_version_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MapsPlatformDatasetsV1AlphaClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = MapsPlatformDatasetsV1AlphaClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = MapsPlatformDatasetsV1AlphaClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MapsPlatformDatasetsV1AlphaClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = MapsPlatformDatasetsV1AlphaClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, transports.MapsPlatformDatasetsV1AlphaRestTransport])
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
        return 10
    transport = MapsPlatformDatasetsV1AlphaClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.MapsPlatformDatasetsV1AlphaGrpcTransport)

def test_maps_platform_datasets_v1_alpha_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.MapsPlatformDatasetsV1AlphaTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_maps_platform_datasets_v1_alpha_base_transport():
    if False:
        return 10
    with mock.patch('google.maps.mapsplatformdatasets_v1alpha.services.maps_platform_datasets_v1_alpha.transports.MapsPlatformDatasetsV1AlphaTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.MapsPlatformDatasetsV1AlphaTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_dataset', 'update_dataset_metadata', 'get_dataset', 'list_dataset_versions', 'list_datasets', 'delete_dataset', 'delete_dataset_version')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_maps_platform_datasets_v1_alpha_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.maps.mapsplatformdatasets_v1alpha.services.maps_platform_datasets_v1_alpha.transports.MapsPlatformDatasetsV1AlphaTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.MapsPlatformDatasetsV1AlphaTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_maps_platform_datasets_v1_alpha_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.maps.mapsplatformdatasets_v1alpha.services.maps_platform_datasets_v1_alpha.transports.MapsPlatformDatasetsV1AlphaTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.MapsPlatformDatasetsV1AlphaTransport()
        adc.assert_called_once()

def test_maps_platform_datasets_v1_alpha_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        MapsPlatformDatasetsV1AlphaClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport])
def test_maps_platform_datasets_v1_alpha_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, transports.MapsPlatformDatasetsV1AlphaRestTransport])
def test_maps_platform_datasets_v1_alpha_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.MapsPlatformDatasetsV1AlphaGrpcTransport, grpc_helpers), (transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport, grpc_helpers_async)])
def test_maps_platform_datasets_v1_alpha_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('mapsplatformdatasets.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='mapsplatformdatasets.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport])
def test_maps_platform_datasets_v1_alpha_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_maps_platform_datasets_v1_alpha_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.MapsPlatformDatasetsV1AlphaRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_maps_platform_datasets_v1_alpha_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='mapsplatformdatasets.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('mapsplatformdatasets.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://mapsplatformdatasets.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_maps_platform_datasets_v1_alpha_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='mapsplatformdatasets.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('mapsplatformdatasets.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://mapsplatformdatasets.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_maps_platform_datasets_v1_alpha_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = MapsPlatformDatasetsV1AlphaClient(credentials=creds1, transport=transport_name)
    client2 = MapsPlatformDatasetsV1AlphaClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_dataset._session
    session2 = client2.transport.create_dataset._session
    assert session1 != session2
    session1 = client1.transport.update_dataset_metadata._session
    session2 = client2.transport.update_dataset_metadata._session
    assert session1 != session2
    session1 = client1.transport.get_dataset._session
    session2 = client2.transport.get_dataset._session
    assert session1 != session2
    session1 = client1.transport.list_dataset_versions._session
    session2 = client2.transport.list_dataset_versions._session
    assert session1 != session2
    session1 = client1.transport.list_datasets._session
    session2 = client2.transport.list_datasets._session
    assert session1 != session2
    session1 = client1.transport.delete_dataset._session
    session2 = client2.transport.delete_dataset._session
    assert session1 != session2
    session1 = client1.transport.delete_dataset_version._session
    session2 = client2.transport.delete_dataset_version._session
    assert session1 != session2

def test_maps_platform_datasets_v1_alpha_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_maps_platform_datasets_v1_alpha_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport])
def test_maps_platform_datasets_v1_alpha_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.MapsPlatformDatasetsV1AlphaGrpcTransport, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport])
def test_maps_platform_datasets_v1_alpha_transport_channel_mtls_with_adc(transport_class):
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

def test_dataset_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    dataset = 'clam'
    expected = 'projects/{project}/datasets/{dataset}'.format(project=project, dataset=dataset)
    actual = MapsPlatformDatasetsV1AlphaClient.dataset_path(project, dataset)
    assert expected == actual

def test_parse_dataset_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'dataset': 'octopus'}
    path = MapsPlatformDatasetsV1AlphaClient.dataset_path(**expected)
    actual = MapsPlatformDatasetsV1AlphaClient.parse_dataset_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = MapsPlatformDatasetsV1AlphaClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'nudibranch'}
    path = MapsPlatformDatasetsV1AlphaClient.common_billing_account_path(**expected)
    actual = MapsPlatformDatasetsV1AlphaClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = MapsPlatformDatasetsV1AlphaClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'mussel'}
    path = MapsPlatformDatasetsV1AlphaClient.common_folder_path(**expected)
    actual = MapsPlatformDatasetsV1AlphaClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = MapsPlatformDatasetsV1AlphaClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nautilus'}
    path = MapsPlatformDatasetsV1AlphaClient.common_organization_path(**expected)
    actual = MapsPlatformDatasetsV1AlphaClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = MapsPlatformDatasetsV1AlphaClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone'}
    path = MapsPlatformDatasetsV1AlphaClient.common_project_path(**expected)
    actual = MapsPlatformDatasetsV1AlphaClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = MapsPlatformDatasetsV1AlphaClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = MapsPlatformDatasetsV1AlphaClient.common_location_path(**expected)
    actual = MapsPlatformDatasetsV1AlphaClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.MapsPlatformDatasetsV1AlphaTransport, '_prep_wrapped_messages') as prep:
        client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.MapsPlatformDatasetsV1AlphaTransport, '_prep_wrapped_messages') as prep:
        transport_class = MapsPlatformDatasetsV1AlphaClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = MapsPlatformDatasetsV1AlphaAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        while True:
            i = 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = MapsPlatformDatasetsV1AlphaClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(MapsPlatformDatasetsV1AlphaClient, transports.MapsPlatformDatasetsV1AlphaGrpcTransport), (MapsPlatformDatasetsV1AlphaAsyncClient, transports.MapsPlatformDatasetsV1AlphaGrpcAsyncIOTransport)])
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