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
from google.cloud.bigquery_biglake_v1alpha1.services.metastore_service import MetastoreServiceAsyncClient, MetastoreServiceClient, pagers, transports
from google.cloud.bigquery_biglake_v1alpha1.types import metastore

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert MetastoreServiceClient._get_default_mtls_endpoint(None) is None
    assert MetastoreServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert MetastoreServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert MetastoreServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert MetastoreServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert MetastoreServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(MetastoreServiceClient, 'grpc'), (MetastoreServiceAsyncClient, 'grpc_asyncio'), (MetastoreServiceClient, 'rest')])
def test_metastore_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('biglake.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://biglake.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.MetastoreServiceGrpcTransport, 'grpc'), (transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.MetastoreServiceRestTransport, 'rest')])
def test_metastore_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(MetastoreServiceClient, 'grpc'), (MetastoreServiceAsyncClient, 'grpc_asyncio'), (MetastoreServiceClient, 'rest')])
def test_metastore_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('biglake.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://biglake.googleapis.com')

def test_metastore_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = MetastoreServiceClient.get_transport_class()
    available_transports = [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceRestTransport]
    assert transport in available_transports
    transport = MetastoreServiceClient.get_transport_class('grpc')
    assert transport == transports.MetastoreServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(MetastoreServiceClient, transports.MetastoreServiceGrpcTransport, 'grpc'), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (MetastoreServiceClient, transports.MetastoreServiceRestTransport, 'rest')])
@mock.patch.object(MetastoreServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetastoreServiceClient))
@mock.patch.object(MetastoreServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetastoreServiceAsyncClient))
def test_metastore_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(MetastoreServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(MetastoreServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(MetastoreServiceClient, transports.MetastoreServiceGrpcTransport, 'grpc', 'true'), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (MetastoreServiceClient, transports.MetastoreServiceGrpcTransport, 'grpc', 'false'), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (MetastoreServiceClient, transports.MetastoreServiceRestTransport, 'rest', 'true'), (MetastoreServiceClient, transports.MetastoreServiceRestTransport, 'rest', 'false')])
@mock.patch.object(MetastoreServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetastoreServiceClient))
@mock.patch.object(MetastoreServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetastoreServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_metastore_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [MetastoreServiceClient, MetastoreServiceAsyncClient])
@mock.patch.object(MetastoreServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetastoreServiceClient))
@mock.patch.object(MetastoreServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(MetastoreServiceAsyncClient))
def test_metastore_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(MetastoreServiceClient, transports.MetastoreServiceGrpcTransport, 'grpc'), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (MetastoreServiceClient, transports.MetastoreServiceRestTransport, 'rest')])
def test_metastore_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(MetastoreServiceClient, transports.MetastoreServiceGrpcTransport, 'grpc', grpc_helpers), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (MetastoreServiceClient, transports.MetastoreServiceRestTransport, 'rest', None)])
def test_metastore_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_metastore_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.bigquery_biglake_v1alpha1.services.metastore_service.transports.MetastoreServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = MetastoreServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(MetastoreServiceClient, transports.MetastoreServiceGrpcTransport, 'grpc', grpc_helpers), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_metastore_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
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
        create_channel.assert_called_with('biglake.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), scopes=None, default_host='biglake.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [metastore.CreateCatalogRequest, dict])
def test_create_catalog(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        call.return_value = metastore.Catalog(name='name_value')
        response = client.create_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateCatalogRequest()
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

def test_create_catalog_empty_call():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        client.create_catalog()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateCatalogRequest()

@pytest.mark.asyncio
async def test_create_catalog_async(transport: str='grpc_asyncio', request_type=metastore.CreateCatalogRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog(name='name_value'))
        response = await client.create_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateCatalogRequest()
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_catalog_async_from_dict():
    await test_create_catalog_async(request_type=dict)

def test_create_catalog_field_headers():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateCatalogRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        client.create_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_catalog_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateCatalogRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog())
        await client.create_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_catalog_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        client.create_catalog(parent='parent_value', catalog=metastore.Catalog(name='name_value'), catalog_id='catalog_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].catalog
        mock_val = metastore.Catalog(name='name_value')
        assert arg == mock_val
        arg = args[0].catalog_id
        mock_val = 'catalog_id_value'
        assert arg == mock_val

def test_create_catalog_flattened_error():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_catalog(metastore.CreateCatalogRequest(), parent='parent_value', catalog=metastore.Catalog(name='name_value'), catalog_id='catalog_id_value')

@pytest.mark.asyncio
async def test_create_catalog_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog())
        response = await client.create_catalog(parent='parent_value', catalog=metastore.Catalog(name='name_value'), catalog_id='catalog_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].catalog
        mock_val = metastore.Catalog(name='name_value')
        assert arg == mock_val
        arg = args[0].catalog_id
        mock_val = 'catalog_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_catalog_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_catalog(metastore.CreateCatalogRequest(), parent='parent_value', catalog=metastore.Catalog(name='name_value'), catalog_id='catalog_id_value')

@pytest.mark.parametrize('request_type', [metastore.DeleteCatalogRequest, dict])
def test_delete_catalog(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        call.return_value = metastore.Catalog(name='name_value')
        response = client.delete_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteCatalogRequest()
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

def test_delete_catalog_empty_call():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        client.delete_catalog()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteCatalogRequest()

@pytest.mark.asyncio
async def test_delete_catalog_async(transport: str='grpc_asyncio', request_type=metastore.DeleteCatalogRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog(name='name_value'))
        response = await client.delete_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteCatalogRequest()
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_delete_catalog_async_from_dict():
    await test_delete_catalog_async(request_type=dict)

def test_delete_catalog_field_headers():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteCatalogRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        client.delete_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_catalog_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteCatalogRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog())
        await client.delete_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_catalog_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        client.delete_catalog(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_catalog_flattened_error():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_catalog(metastore.DeleteCatalogRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_catalog_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog())
        response = await client.delete_catalog(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_catalog_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_catalog(metastore.DeleteCatalogRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.GetCatalogRequest, dict])
def test_get_catalog(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        call.return_value = metastore.Catalog(name='name_value')
        response = client.get_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetCatalogRequest()
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

def test_get_catalog_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        client.get_catalog()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetCatalogRequest()

@pytest.mark.asyncio
async def test_get_catalog_async(transport: str='grpc_asyncio', request_type=metastore.GetCatalogRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog(name='name_value'))
        response = await client.get_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetCatalogRequest()
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_catalog_async_from_dict():
    await test_get_catalog_async(request_type=dict)

def test_get_catalog_field_headers():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.GetCatalogRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        client.get_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_catalog_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.GetCatalogRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog())
        await client.get_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_catalog_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        client.get_catalog(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_catalog_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_catalog(metastore.GetCatalogRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_catalog_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_catalog), '__call__') as call:
        call.return_value = metastore.Catalog()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Catalog())
        response = await client.get_catalog(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_catalog_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_catalog(metastore.GetCatalogRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.ListCatalogsRequest, dict])
def test_list_catalogs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = metastore.ListCatalogsResponse(next_page_token='next_page_token_value')
        response = client.list_catalogs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListCatalogsRequest()
    assert isinstance(response, pagers.ListCatalogsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_catalogs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        client.list_catalogs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListCatalogsRequest()

@pytest.mark.asyncio
async def test_list_catalogs_async(transport: str='grpc_asyncio', request_type=metastore.ListCatalogsRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListCatalogsResponse(next_page_token='next_page_token_value'))
        response = await client.list_catalogs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListCatalogsRequest()
    assert isinstance(response, pagers.ListCatalogsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_catalogs_async_from_dict():
    await test_list_catalogs_async(request_type=dict)

def test_list_catalogs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListCatalogsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = metastore.ListCatalogsResponse()
        client.list_catalogs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_catalogs_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListCatalogsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListCatalogsResponse())
        await client.list_catalogs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_catalogs_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = metastore.ListCatalogsResponse()
        client.list_catalogs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_catalogs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_catalogs(metastore.ListCatalogsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_catalogs_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = metastore.ListCatalogsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListCatalogsResponse())
        response = await client.list_catalogs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_catalogs_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_catalogs(metastore.ListCatalogsRequest(), parent='parent_value')

def test_list_catalogs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.side_effect = (metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog(), metastore.Catalog()], next_page_token='abc'), metastore.ListCatalogsResponse(catalogs=[], next_page_token='def'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog()], next_page_token='ghi'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_catalogs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Catalog) for i in results))

def test_list_catalogs_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.side_effect = (metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog(), metastore.Catalog()], next_page_token='abc'), metastore.ListCatalogsResponse(catalogs=[], next_page_token='def'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog()], next_page_token='ghi'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog()]), RuntimeError)
        pages = list(client.list_catalogs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_catalogs_async_pager():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog(), metastore.Catalog()], next_page_token='abc'), metastore.ListCatalogsResponse(catalogs=[], next_page_token='def'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog()], next_page_token='ghi'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog()]), RuntimeError)
        async_pager = await client.list_catalogs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, metastore.Catalog) for i in responses))

@pytest.mark.asyncio
async def test_list_catalogs_async_pages():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog(), metastore.Catalog()], next_page_token='abc'), metastore.ListCatalogsResponse(catalogs=[], next_page_token='def'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog()], next_page_token='ghi'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_catalogs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateDatabaseRequest, dict])
def test_create_database(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        call.return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response = client.create_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_create_database_empty_call():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        client.create_database()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateDatabaseRequest()

@pytest.mark.asyncio
async def test_create_database_async(transport: str='grpc_asyncio', request_type=metastore.CreateDatabaseRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE))
        response = await client.create_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

@pytest.mark.asyncio
async def test_create_database_async_from_dict():
    await test_create_database_async(request_type=dict)

def test_create_database_field_headers():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateDatabaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.create_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_database_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateDatabaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        await client.create_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_database_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.create_database(parent='parent_value', database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), database_id='database_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].database
        mock_val = metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value'))
        assert arg == mock_val
        arg = args[0].database_id
        mock_val = 'database_id_value'
        assert arg == mock_val

def test_create_database_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_database(metastore.CreateDatabaseRequest(), parent='parent_value', database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), database_id='database_id_value')

@pytest.mark.asyncio
async def test_create_database_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_database), '__call__') as call:
        call.return_value = metastore.Database()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        response = await client.create_database(parent='parent_value', database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), database_id='database_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].database
        mock_val = metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value'))
        assert arg == mock_val
        arg = args[0].database_id
        mock_val = 'database_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_database_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_database(metastore.CreateDatabaseRequest(), parent='parent_value', database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), database_id='database_id_value')

@pytest.mark.parametrize('request_type', [metastore.DeleteDatabaseRequest, dict])
def test_delete_database(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        call.return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response = client.delete_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_delete_database_empty_call():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        client.delete_database()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteDatabaseRequest()

@pytest.mark.asyncio
async def test_delete_database_async(transport: str='grpc_asyncio', request_type=metastore.DeleteDatabaseRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE))
        response = await client.delete_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

@pytest.mark.asyncio
async def test_delete_database_async_from_dict():
    await test_delete_database_async(request_type=dict)

def test_delete_database_field_headers():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteDatabaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.delete_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_database_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteDatabaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        await client.delete_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_database_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.delete_database(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_database_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_database(metastore.DeleteDatabaseRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_database_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_database), '__call__') as call:
        call.return_value = metastore.Database()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        response = await client.delete_database(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_database_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_database(metastore.DeleteDatabaseRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.UpdateDatabaseRequest, dict])
def test_update_database(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        call.return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response = client.update_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.UpdateDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_update_database_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        client.update_database()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.UpdateDatabaseRequest()

@pytest.mark.asyncio
async def test_update_database_async(transport: str='grpc_asyncio', request_type=metastore.UpdateDatabaseRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE))
        response = await client.update_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.UpdateDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

@pytest.mark.asyncio
async def test_update_database_async_from_dict():
    await test_update_database_async(request_type=dict)

def test_update_database_field_headers():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.UpdateDatabaseRequest()
    request.database.name = 'name_value'
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.update_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'database.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_database_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.UpdateDatabaseRequest()
    request.database.name = 'name_value'
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        await client.update_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'database.name=name_value') in kw['metadata']

def test_update_database_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.update_database(database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].database
        mock_val = metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_database_flattened_error():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_database(metastore.UpdateDatabaseRequest(), database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_database_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_database), '__call__') as call:
        call.return_value = metastore.Database()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        response = await client.update_database(database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].database
        mock_val = metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_database_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_database(metastore.UpdateDatabaseRequest(), database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [metastore.GetDatabaseRequest, dict])
def test_get_database(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        call.return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response = client.get_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_get_database_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        client.get_database()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetDatabaseRequest()

@pytest.mark.asyncio
async def test_get_database_async(transport: str='grpc_asyncio', request_type=metastore.GetDatabaseRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE))
        response = await client.get_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetDatabaseRequest()
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

@pytest.mark.asyncio
async def test_get_database_async_from_dict():
    await test_get_database_async(request_type=dict)

def test_get_database_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.GetDatabaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.get_database(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_database_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.GetDatabaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        await client.get_database(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_database_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        call.return_value = metastore.Database()
        client.get_database(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_database_flattened_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_database(metastore.GetDatabaseRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_database_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_database), '__call__') as call:
        call.return_value = metastore.Database()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Database())
        response = await client.get_database(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_database_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_database(metastore.GetDatabaseRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.ListDatabasesRequest, dict])
def test_list_databases(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.return_value = metastore.ListDatabasesResponse(next_page_token='next_page_token_value')
        response = client.list_databases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListDatabasesRequest()
    assert isinstance(response, pagers.ListDatabasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_databases_empty_call():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        client.list_databases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListDatabasesRequest()

@pytest.mark.asyncio
async def test_list_databases_async(transport: str='grpc_asyncio', request_type=metastore.ListDatabasesRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListDatabasesResponse(next_page_token='next_page_token_value'))
        response = await client.list_databases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListDatabasesRequest()
    assert isinstance(response, pagers.ListDatabasesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_databases_async_from_dict():
    await test_list_databases_async(request_type=dict)

def test_list_databases_field_headers():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListDatabasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.return_value = metastore.ListDatabasesResponse()
        client.list_databases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_databases_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListDatabasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListDatabasesResponse())
        await client.list_databases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_databases_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.return_value = metastore.ListDatabasesResponse()
        client.list_databases(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_databases_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_databases(metastore.ListDatabasesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_databases_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.return_value = metastore.ListDatabasesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListDatabasesResponse())
        response = await client.list_databases(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_databases_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_databases(metastore.ListDatabasesRequest(), parent='parent_value')

def test_list_databases_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.side_effect = (metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database(), metastore.Database()], next_page_token='abc'), metastore.ListDatabasesResponse(databases=[], next_page_token='def'), metastore.ListDatabasesResponse(databases=[metastore.Database()], next_page_token='ghi'), metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_databases(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Database) for i in results))

def test_list_databases_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_databases), '__call__') as call:
        call.side_effect = (metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database(), metastore.Database()], next_page_token='abc'), metastore.ListDatabasesResponse(databases=[], next_page_token='def'), metastore.ListDatabasesResponse(databases=[metastore.Database()], next_page_token='ghi'), metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database()]), RuntimeError)
        pages = list(client.list_databases(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_databases_async_pager():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_databases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database(), metastore.Database()], next_page_token='abc'), metastore.ListDatabasesResponse(databases=[], next_page_token='def'), metastore.ListDatabasesResponse(databases=[metastore.Database()], next_page_token='ghi'), metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database()]), RuntimeError)
        async_pager = await client.list_databases(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, metastore.Database) for i in responses))

@pytest.mark.asyncio
async def test_list_databases_async_pages():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_databases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database(), metastore.Database()], next_page_token='abc'), metastore.ListDatabasesResponse(databases=[], next_page_token='def'), metastore.ListDatabasesResponse(databases=[metastore.Database()], next_page_token='ghi'), metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_databases(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateTableRequest, dict])
def test_create_table(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        call.return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response = client.create_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_create_table_empty_call():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        client.create_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateTableRequest()

@pytest.mark.asyncio
async def test_create_table_async(transport: str='grpc_asyncio', request_type=metastore.CreateTableRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value'))
        response = await client.create_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_create_table_async_from_dict():
    await test_create_table_async(request_type=dict)

def test_create_table_field_headers():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateTableRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.create_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_table_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateTableRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        await client.create_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_table_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.create_table(parent='parent_value', table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), table_id='table_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].table
        mock_val = metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'}))
        assert arg == mock_val
        arg = args[0].table_id
        mock_val = 'table_id_value'
        assert arg == mock_val

def test_create_table_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_table(metastore.CreateTableRequest(), parent='parent_value', table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), table_id='table_id_value')

@pytest.mark.asyncio
async def test_create_table_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_table), '__call__') as call:
        call.return_value = metastore.Table()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        response = await client.create_table(parent='parent_value', table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), table_id='table_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].table
        mock_val = metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'}))
        assert arg == mock_val
        arg = args[0].table_id
        mock_val = 'table_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_table_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_table(metastore.CreateTableRequest(), parent='parent_value', table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), table_id='table_id_value')

@pytest.mark.parametrize('request_type', [metastore.DeleteTableRequest, dict])
def test_delete_table(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        call.return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response = client.delete_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_delete_table_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        client.delete_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteTableRequest()

@pytest.mark.asyncio
async def test_delete_table_async(transport: str='grpc_asyncio', request_type=metastore.DeleteTableRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value'))
        response = await client.delete_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_delete_table_async_from_dict():
    await test_delete_table_async(request_type=dict)

def test_delete_table_field_headers():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.delete_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_table_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        await client.delete_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_table_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.delete_table(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_table_flattened_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_table(metastore.DeleteTableRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_table_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_table), '__call__') as call:
        call.return_value = metastore.Table()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        response = await client.delete_table(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_table_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_table(metastore.DeleteTableRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.UpdateTableRequest, dict])
def test_update_table(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        call.return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response = client.update_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.UpdateTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_update_table_empty_call():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        client.update_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.UpdateTableRequest()

@pytest.mark.asyncio
async def test_update_table_async(transport: str='grpc_asyncio', request_type=metastore.UpdateTableRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value'))
        response = await client.update_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.UpdateTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_update_table_async_from_dict():
    await test_update_table_async(request_type=dict)

def test_update_table_field_headers():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.UpdateTableRequest()
    request.table.name = 'name_value'
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.update_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'table.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_table_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.UpdateTableRequest()
    request.table.name = 'name_value'
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        await client.update_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'table.name=name_value') in kw['metadata']

def test_update_table_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.update_table(table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].table
        mock_val = metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'}))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_table_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_table(metastore.UpdateTableRequest(), table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_table_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_table), '__call__') as call:
        call.return_value = metastore.Table()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        response = await client.update_table(table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].table
        mock_val = metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'}))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_table_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_table(metastore.UpdateTableRequest(), table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [metastore.RenameTableRequest, dict])
def test_rename_table(request_type, transport: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        call.return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response = client.rename_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.RenameTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_rename_table_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        client.rename_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.RenameTableRequest()

@pytest.mark.asyncio
async def test_rename_table_async(transport: str='grpc_asyncio', request_type=metastore.RenameTableRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value'))
        response = await client.rename_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.RenameTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_rename_table_async_from_dict():
    await test_rename_table_async(request_type=dict)

def test_rename_table_field_headers():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.RenameTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.rename_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_rename_table_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.RenameTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        await client.rename_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_rename_table_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.rename_table(name='name_value', new_name='new_name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_name
        mock_val = 'new_name_value'
        assert arg == mock_val

def test_rename_table_flattened_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.rename_table(metastore.RenameTableRequest(), name='name_value', new_name='new_name_value')

@pytest.mark.asyncio
async def test_rename_table_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_table), '__call__') as call:
        call.return_value = metastore.Table()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        response = await client.rename_table(name='name_value', new_name='new_name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_name
        mock_val = 'new_name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_rename_table_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.rename_table(metastore.RenameTableRequest(), name='name_value', new_name='new_name_value')

@pytest.mark.parametrize('request_type', [metastore.GetTableRequest, dict])
def test_get_table(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response = client.get_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_get_table_empty_call():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        client.get_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetTableRequest()

@pytest.mark.asyncio
async def test_get_table_async(transport: str='grpc_asyncio', request_type=metastore.GetTableRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value'))
        response = await client.get_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.GetTableRequest()
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_table_async_from_dict():
    await test_get_table_async(request_type=dict)

def test_get_table_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.GetTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.get_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_table_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.GetTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        await client.get_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_table_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = metastore.Table()
        client.get_table(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_table_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_table(metastore.GetTableRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_table_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = metastore.Table()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Table())
        response = await client.get_table(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_table_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_table(metastore.GetTableRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.ListTablesRequest, dict])
def test_list_tables(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = metastore.ListTablesResponse(next_page_token='next_page_token_value')
        response = client.list_tables(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListTablesRequest()
    assert isinstance(response, pagers.ListTablesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tables_empty_call():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        client.list_tables()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListTablesRequest()

@pytest.mark.asyncio
async def test_list_tables_async(transport: str='grpc_asyncio', request_type=metastore.ListTablesRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListTablesResponse(next_page_token='next_page_token_value'))
        response = await client.list_tables(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListTablesRequest()
    assert isinstance(response, pagers.ListTablesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tables_async_from_dict():
    await test_list_tables_async(request_type=dict)

def test_list_tables_field_headers():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListTablesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = metastore.ListTablesResponse()
        client.list_tables(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_tables_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListTablesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListTablesResponse())
        await client.list_tables(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_tables_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = metastore.ListTablesResponse()
        client.list_tables(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tables_flattened_error():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tables(metastore.ListTablesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tables_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = metastore.ListTablesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListTablesResponse())
        response = await client.list_tables(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tables_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tables(metastore.ListTablesRequest(), parent='parent_value')

def test_list_tables_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.side_effect = (metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table(), metastore.Table()], next_page_token='abc'), metastore.ListTablesResponse(tables=[], next_page_token='def'), metastore.ListTablesResponse(tables=[metastore.Table()], next_page_token='ghi'), metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_tables(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Table) for i in results))

def test_list_tables_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.side_effect = (metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table(), metastore.Table()], next_page_token='abc'), metastore.ListTablesResponse(tables=[], next_page_token='def'), metastore.ListTablesResponse(tables=[metastore.Table()], next_page_token='ghi'), metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table()]), RuntimeError)
        pages = list(client.list_tables(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tables_async_pager():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tables), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table(), metastore.Table()], next_page_token='abc'), metastore.ListTablesResponse(tables=[], next_page_token='def'), metastore.ListTablesResponse(tables=[metastore.Table()], next_page_token='ghi'), metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table()]), RuntimeError)
        async_pager = await client.list_tables(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, metastore.Table) for i in responses))

@pytest.mark.asyncio
async def test_list_tables_async_pages():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tables), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table(), metastore.Table()], next_page_token='abc'), metastore.ListTablesResponse(tables=[], next_page_token='def'), metastore.ListTablesResponse(tables=[metastore.Table()], next_page_token='ghi'), metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tables(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateLockRequest, dict])
def test_create_lock(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        call.return_value = metastore.Lock(name='name_value', type_=metastore.Lock.Type.EXCLUSIVE, state=metastore.Lock.State.WAITING, table_id='table_id_value')
        response = client.create_lock(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateLockRequest()
    assert isinstance(response, metastore.Lock)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Lock.Type.EXCLUSIVE
    assert response.state == metastore.Lock.State.WAITING

def test_create_lock_empty_call():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        client.create_lock()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateLockRequest()

@pytest.mark.asyncio
async def test_create_lock_async(transport: str='grpc_asyncio', request_type=metastore.CreateLockRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Lock(name='name_value', type_=metastore.Lock.Type.EXCLUSIVE, state=metastore.Lock.State.WAITING))
        response = await client.create_lock(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CreateLockRequest()
    assert isinstance(response, metastore.Lock)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Lock.Type.EXCLUSIVE
    assert response.state == metastore.Lock.State.WAITING

@pytest.mark.asyncio
async def test_create_lock_async_from_dict():
    await test_create_lock_async(request_type=dict)

def test_create_lock_field_headers():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateLockRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        call.return_value = metastore.Lock()
        client.create_lock(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_lock_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CreateLockRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Lock())
        await client.create_lock(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_lock_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        call.return_value = metastore.Lock()
        client.create_lock(parent='parent_value', lock=metastore.Lock(table_id='table_id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].lock
        mock_val = metastore.Lock(table_id='table_id_value')
        assert arg == mock_val

def test_create_lock_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_lock(metastore.CreateLockRequest(), parent='parent_value', lock=metastore.Lock(table_id='table_id_value'))

@pytest.mark.asyncio
async def test_create_lock_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_lock), '__call__') as call:
        call.return_value = metastore.Lock()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Lock())
        response = await client.create_lock(parent='parent_value', lock=metastore.Lock(table_id='table_id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].lock
        mock_val = metastore.Lock(table_id='table_id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_lock_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_lock(metastore.CreateLockRequest(), parent='parent_value', lock=metastore.Lock(table_id='table_id_value'))

@pytest.mark.parametrize('request_type', [metastore.DeleteLockRequest, dict])
def test_delete_lock(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        call.return_value = None
        response = client.delete_lock(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteLockRequest()
    assert response is None

def test_delete_lock_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        client.delete_lock()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteLockRequest()

@pytest.mark.asyncio
async def test_delete_lock_async(transport: str='grpc_asyncio', request_type=metastore.DeleteLockRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_lock(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.DeleteLockRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_lock_async_from_dict():
    await test_delete_lock_async(request_type=dict)

def test_delete_lock_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteLockRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        call.return_value = None
        client.delete_lock(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_lock_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.DeleteLockRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_lock(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_lock_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        call.return_value = None
        client.delete_lock(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_lock_flattened_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_lock(metastore.DeleteLockRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_lock_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_lock), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_lock(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_lock_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_lock(metastore.DeleteLockRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.CheckLockRequest, dict])
def test_check_lock(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        call.return_value = metastore.Lock(name='name_value', type_=metastore.Lock.Type.EXCLUSIVE, state=metastore.Lock.State.WAITING, table_id='table_id_value')
        response = client.check_lock(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CheckLockRequest()
    assert isinstance(response, metastore.Lock)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Lock.Type.EXCLUSIVE
    assert response.state == metastore.Lock.State.WAITING

def test_check_lock_empty_call():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        client.check_lock()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CheckLockRequest()

@pytest.mark.asyncio
async def test_check_lock_async(transport: str='grpc_asyncio', request_type=metastore.CheckLockRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Lock(name='name_value', type_=metastore.Lock.Type.EXCLUSIVE, state=metastore.Lock.State.WAITING))
        response = await client.check_lock(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.CheckLockRequest()
    assert isinstance(response, metastore.Lock)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Lock.Type.EXCLUSIVE
    assert response.state == metastore.Lock.State.WAITING

@pytest.mark.asyncio
async def test_check_lock_async_from_dict():
    await test_check_lock_async(request_type=dict)

def test_check_lock_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CheckLockRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        call.return_value = metastore.Lock()
        client.check_lock(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_check_lock_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.CheckLockRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Lock())
        await client.check_lock(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_check_lock_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        call.return_value = metastore.Lock()
        client.check_lock(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_check_lock_flattened_error():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.check_lock(metastore.CheckLockRequest(), name='name_value')

@pytest.mark.asyncio
async def test_check_lock_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.check_lock), '__call__') as call:
        call.return_value = metastore.Lock()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.Lock())
        response = await client.check_lock(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_check_lock_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.check_lock(metastore.CheckLockRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [metastore.ListLocksRequest, dict])
def test_list_locks(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.return_value = metastore.ListLocksResponse(next_page_token='next_page_token_value')
        response = client.list_locks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListLocksRequest()
    assert isinstance(response, pagers.ListLocksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_locks_empty_call():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        client.list_locks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListLocksRequest()

@pytest.mark.asyncio
async def test_list_locks_async(transport: str='grpc_asyncio', request_type=metastore.ListLocksRequest):
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListLocksResponse(next_page_token='next_page_token_value'))
        response = await client.list_locks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == metastore.ListLocksRequest()
    assert isinstance(response, pagers.ListLocksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_locks_async_from_dict():
    await test_list_locks_async(request_type=dict)

def test_list_locks_field_headers():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListLocksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.return_value = metastore.ListLocksResponse()
        client.list_locks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_locks_field_headers_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = metastore.ListLocksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListLocksResponse())
        await client.list_locks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_locks_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.return_value = metastore.ListLocksResponse()
        client.list_locks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_locks_flattened_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_locks(metastore.ListLocksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_locks_flattened_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.return_value = metastore.ListLocksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(metastore.ListLocksResponse())
        response = await client.list_locks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_locks_flattened_error_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_locks(metastore.ListLocksRequest(), parent='parent_value')

def test_list_locks_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.side_effect = (metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock(), metastore.Lock()], next_page_token='abc'), metastore.ListLocksResponse(locks=[], next_page_token='def'), metastore.ListLocksResponse(locks=[metastore.Lock()], next_page_token='ghi'), metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_locks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Lock) for i in results))

def test_list_locks_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_locks), '__call__') as call:
        call.side_effect = (metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock(), metastore.Lock()], next_page_token='abc'), metastore.ListLocksResponse(locks=[], next_page_token='def'), metastore.ListLocksResponse(locks=[metastore.Lock()], next_page_token='ghi'), metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock()]), RuntimeError)
        pages = list(client.list_locks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_locks_async_pager():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_locks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock(), metastore.Lock()], next_page_token='abc'), metastore.ListLocksResponse(locks=[], next_page_token='def'), metastore.ListLocksResponse(locks=[metastore.Lock()], next_page_token='ghi'), metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock()]), RuntimeError)
        async_pager = await client.list_locks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, metastore.Lock) for i in responses))

@pytest.mark.asyncio
async def test_list_locks_async_pages():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_locks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock(), metastore.Lock()], next_page_token='abc'), metastore.ListLocksResponse(locks=[], next_page_token='def'), metastore.ListLocksResponse(locks=[metastore.Lock()], next_page_token='ghi'), metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_locks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateCatalogRequest, dict])
def test_create_catalog_rest(request_type):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['catalog'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}}
    test_field = metastore.CreateCatalogRequest.meta.fields['catalog']

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
    for (field, value) in request_init['catalog'].items():
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
                for i in range(0, len(request_init['catalog'][field])):
                    del request_init['catalog'][field][i][subfield]
            else:
                del request_init['catalog'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Catalog(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_catalog(request)
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

def test_create_catalog_rest_required_fields(request_type=metastore.CreateCatalogRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['catalog_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'catalogId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_catalog._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'catalogId' in jsonified_request
    assert jsonified_request['catalogId'] == request_init['catalog_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['catalogId'] = 'catalog_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_catalog._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('catalog_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'catalogId' in jsonified_request
    assert jsonified_request['catalogId'] == 'catalog_id_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Catalog()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Catalog.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_catalog(request)
            expected_params = [('catalogId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_catalog_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_catalog._get_unset_required_fields({})
    assert set(unset_fields) == set(('catalogId',)) & set(('parent', 'catalog', 'catalogId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_catalog_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_create_catalog') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_create_catalog') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.CreateCatalogRequest.pb(metastore.CreateCatalogRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Catalog.to_json(metastore.Catalog())
        request = metastore.CreateCatalogRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Catalog()
        client.create_catalog(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_catalog_rest_bad_request(transport: str='rest', request_type=metastore.CreateCatalogRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_catalog(request)

def test_create_catalog_rest_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Catalog()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', catalog=metastore.Catalog(name='name_value'), catalog_id='catalog_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_catalog(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*}/catalogs' % client.transport._host, args[1])

def test_create_catalog_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_catalog(metastore.CreateCatalogRequest(), parent='parent_value', catalog=metastore.Catalog(name='name_value'), catalog_id='catalog_id_value')

def test_create_catalog_rest_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.DeleteCatalogRequest, dict])
def test_delete_catalog_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Catalog(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_catalog(request)
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

def test_delete_catalog_rest_required_fields(request_type=metastore.DeleteCatalogRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_catalog._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_catalog._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Catalog()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Catalog.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_catalog(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_catalog_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_catalog._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_catalog_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_delete_catalog') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_delete_catalog') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.DeleteCatalogRequest.pb(metastore.DeleteCatalogRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Catalog.to_json(metastore.Catalog())
        request = metastore.DeleteCatalogRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Catalog()
        client.delete_catalog(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_catalog_rest_bad_request(transport: str='rest', request_type=metastore.DeleteCatalogRequest):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_catalog(request)

def test_delete_catalog_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Catalog()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_catalog(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*}' % client.transport._host, args[1])

def test_delete_catalog_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_catalog(metastore.DeleteCatalogRequest(), name='name_value')

def test_delete_catalog_rest_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.GetCatalogRequest, dict])
def test_get_catalog_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Catalog(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_catalog(request)
    assert isinstance(response, metastore.Catalog)
    assert response.name == 'name_value'

def test_get_catalog_rest_required_fields(request_type=metastore.GetCatalogRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_catalog._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_catalog._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Catalog()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Catalog.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_catalog(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_catalog_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_catalog._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_catalog_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_get_catalog') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_get_catalog') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.GetCatalogRequest.pb(metastore.GetCatalogRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Catalog.to_json(metastore.Catalog())
        request = metastore.GetCatalogRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Catalog()
        client.get_catalog(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_catalog_rest_bad_request(transport: str='rest', request_type=metastore.GetCatalogRequest):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_catalog(request)

def test_get_catalog_rest_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Catalog()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_catalog(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*}' % client.transport._host, args[1])

def test_get_catalog_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_catalog(metastore.GetCatalogRequest(), name='name_value')

def test_get_catalog_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.ListCatalogsRequest, dict])
def test_list_catalogs_rest(request_type):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListCatalogsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListCatalogsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_catalogs(request)
    assert isinstance(response, pagers.ListCatalogsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_catalogs_rest_required_fields(request_type=metastore.ListCatalogsRequest):
    if False:
        return 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_catalogs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_catalogs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.ListCatalogsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.ListCatalogsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_catalogs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_catalogs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_catalogs._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_catalogs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_list_catalogs') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_list_catalogs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.ListCatalogsRequest.pb(metastore.ListCatalogsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.ListCatalogsResponse.to_json(metastore.ListCatalogsResponse())
        request = metastore.ListCatalogsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.ListCatalogsResponse()
        client.list_catalogs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_catalogs_rest_bad_request(transport: str='rest', request_type=metastore.ListCatalogsRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_catalogs(request)

def test_list_catalogs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListCatalogsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListCatalogsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_catalogs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*}/catalogs' % client.transport._host, args[1])

def test_list_catalogs_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_catalogs(metastore.ListCatalogsRequest(), parent='parent_value')

def test_list_catalogs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog(), metastore.Catalog()], next_page_token='abc'), metastore.ListCatalogsResponse(catalogs=[], next_page_token='def'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog()], next_page_token='ghi'), metastore.ListCatalogsResponse(catalogs=[metastore.Catalog(), metastore.Catalog()]))
        response = response + response
        response = tuple((metastore.ListCatalogsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_catalogs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Catalog) for i in results))
        pages = list(client.list_catalogs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateDatabaseRequest, dict])
def test_create_database_rest(request_type):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request_init['database'] = {'hive_options': {'location_uri': 'location_uri_value', 'parameters': {}}, 'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}, 'type_': 1}
    test_field = metastore.CreateDatabaseRequest.meta.fields['database']

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
    for (field, value) in request_init['database'].items():
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
                for i in range(0, len(request_init['database'][field])):
                    del request_init['database'][field][i][subfield]
            else:
                del request_init['database'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_database(request)
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_create_database_rest_required_fields(request_type=metastore.CreateDatabaseRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['database_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'databaseId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_database._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'databaseId' in jsonified_request
    assert jsonified_request['databaseId'] == request_init['database_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['databaseId'] = 'database_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_database._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('database_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'databaseId' in jsonified_request
    assert jsonified_request['databaseId'] == 'database_id_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Database()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Database.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_database(request)
            expected_params = [('databaseId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_database_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_database._get_unset_required_fields({})
    assert set(unset_fields) == set(('databaseId',)) & set(('parent', 'database', 'databaseId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_database_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_create_database') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_create_database') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.CreateDatabaseRequest.pb(metastore.CreateDatabaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Database.to_json(metastore.Database())
        request = metastore.CreateDatabaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Database()
        client.create_database(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_database_rest_bad_request(transport: str='rest', request_type=metastore.CreateDatabaseRequest):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_database(request)

def test_create_database_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(parent='parent_value', database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), database_id='database_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_database(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*/catalogs/*}/databases' % client.transport._host, args[1])

def test_create_database_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_database(metastore.CreateDatabaseRequest(), parent='parent_value', database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), database_id='database_id_value')

def test_create_database_rest_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.DeleteDatabaseRequest, dict])
def test_delete_database_rest(request_type):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_database(request)
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_delete_database_rest_required_fields(request_type=metastore.DeleteDatabaseRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_database._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_database._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Database()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Database.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_database(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_database_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_database._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_database_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_delete_database') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_delete_database') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.DeleteDatabaseRequest.pb(metastore.DeleteDatabaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Database.to_json(metastore.Database())
        request = metastore.DeleteDatabaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Database()
        client.delete_database(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_database_rest_bad_request(transport: str='rest', request_type=metastore.DeleteDatabaseRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_database(request)

def test_delete_database_rest_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_database(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*}' % client.transport._host, args[1])

def test_delete_database_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_database(metastore.DeleteDatabaseRequest(), name='name_value')

def test_delete_database_rest_error():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.UpdateDatabaseRequest, dict])
def test_update_database_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'database': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}}
    request_init['database'] = {'hive_options': {'location_uri': 'location_uri_value', 'parameters': {}}, 'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}, 'type_': 1}
    test_field = metastore.UpdateDatabaseRequest.meta.fields['database']

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
    for (field, value) in request_init['database'].items():
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
                for i in range(0, len(request_init['database'][field])):
                    del request_init['database'][field][i][subfield]
            else:
                del request_init['database'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_database(request)
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_update_database_rest_required_fields(request_type=metastore.UpdateDatabaseRequest):
    if False:
        return 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_database._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_database._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Database()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Database.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_database(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_database_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_database._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('database',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_database_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_update_database') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_update_database') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.UpdateDatabaseRequest.pb(metastore.UpdateDatabaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Database.to_json(metastore.Database())
        request = metastore.UpdateDatabaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Database()
        client.update_database(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_database_rest_bad_request(transport: str='rest', request_type=metastore.UpdateDatabaseRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'database': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_database(request)

def test_update_database_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database()
        sample_request = {'database': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}}
        mock_args = dict(database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_database(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{database.name=projects/*/locations/*/catalogs/*/databases/*}' % client.transport._host, args[1])

def test_update_database_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_database(metastore.UpdateDatabaseRequest(), database=metastore.Database(hive_options=metastore.HiveDatabaseOptions(location_uri='location_uri_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_database_rest_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.GetDatabaseRequest, dict])
def test_get_database_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database(name='name_value', type_=metastore.Database.Type.HIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_database(request)
    assert isinstance(response, metastore.Database)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Database.Type.HIVE

def test_get_database_rest_required_fields(request_type=metastore.GetDatabaseRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_database._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_database._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Database()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Database.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_database(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_database_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_database._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_database_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_get_database') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_get_database') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.GetDatabaseRequest.pb(metastore.GetDatabaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Database.to_json(metastore.Database())
        request = metastore.GetDatabaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Database()
        client.get_database(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_database_rest_bad_request(transport: str='rest', request_type=metastore.GetDatabaseRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_database(request)

def test_get_database_rest_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Database()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Database.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_database(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*}' % client.transport._host, args[1])

def test_get_database_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_database(metastore.GetDatabaseRequest(), name='name_value')

def test_get_database_rest_error():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.ListDatabasesRequest, dict])
def test_list_databases_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListDatabasesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListDatabasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_databases(request)
    assert isinstance(response, pagers.ListDatabasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_databases_rest_required_fields(request_type=metastore.ListDatabasesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_databases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_databases._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.ListDatabasesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.ListDatabasesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_databases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_databases_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_databases._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_databases_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_list_databases') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_list_databases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.ListDatabasesRequest.pb(metastore.ListDatabasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.ListDatabasesResponse.to_json(metastore.ListDatabasesResponse())
        request = metastore.ListDatabasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.ListDatabasesResponse()
        client.list_databases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_databases_rest_bad_request(transport: str='rest', request_type=metastore.ListDatabasesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_databases(request)

def test_list_databases_rest_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListDatabasesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListDatabasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_databases(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*/catalogs/*}/databases' % client.transport._host, args[1])

def test_list_databases_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_databases(metastore.ListDatabasesRequest(), parent='parent_value')

def test_list_databases_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database(), metastore.Database()], next_page_token='abc'), metastore.ListDatabasesResponse(databases=[], next_page_token='def'), metastore.ListDatabasesResponse(databases=[metastore.Database()], next_page_token='ghi'), metastore.ListDatabasesResponse(databases=[metastore.Database(), metastore.Database()]))
        response = response + response
        response = tuple((metastore.ListDatabasesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
        pager = client.list_databases(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Database) for i in results))
        pages = list(client.list_databases(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateTableRequest, dict])
def test_create_table_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request_init['table'] = {'hive_options': {'parameters': {}, 'table_type': 'table_type_value', 'storage_descriptor': {'location_uri': 'location_uri_value', 'input_format': 'input_format_value', 'output_format': 'output_format_value', 'serde_info': {'serialization_lib': 'serialization_lib_value'}}}, 'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}, 'type_': 1, 'etag': 'etag_value'}
    test_field = metastore.CreateTableRequest.meta.fields['table']

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
    for (field, value) in request_init['table'].items():
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
                for i in range(0, len(request_init['table'][field])):
                    del request_init['table'][field][i][subfield]
            else:
                del request_init['table'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_table(request)
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_create_table_rest_required_fields(request_type=metastore.CreateTableRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['table_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'tableId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'tableId' in jsonified_request
    assert jsonified_request['tableId'] == request_init['table_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['tableId'] = 'table_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_table._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('table_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'tableId' in jsonified_request
    assert jsonified_request['tableId'] == 'table_id_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Table()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Table.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_table(request)
            expected_params = [('tableId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_table_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_table._get_unset_required_fields({})
    assert set(unset_fields) == set(('tableId',)) & set(('parent', 'table', 'tableId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_table_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_create_table') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_create_table') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.CreateTableRequest.pb(metastore.CreateTableRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Table.to_json(metastore.Table())
        request = metastore.CreateTableRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Table()
        client.create_table(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_table_rest_bad_request(transport: str='rest', request_type=metastore.CreateTableRequest):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_table(request)

def test_create_table_rest_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        mock_args = dict(parent='parent_value', table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), table_id='table_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_table(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*/catalogs/*/databases/*}/tables' % client.transport._host, args[1])

def test_create_table_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_table(metastore.CreateTableRequest(), parent='parent_value', table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), table_id='table_id_value')

def test_create_table_rest_error():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.DeleteTableRequest, dict])
def test_delete_table_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_table(request)
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_delete_table_rest_required_fields(request_type=metastore.DeleteTableRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Table()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Table.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_table(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_table_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_table._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_table_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_delete_table') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_delete_table') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.DeleteTableRequest.pb(metastore.DeleteTableRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Table.to_json(metastore.Table())
        request = metastore.DeleteTableRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Table()
        client.delete_table(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_table_rest_bad_request(transport: str='rest', request_type=metastore.DeleteTableRequest):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_table(request)

def test_delete_table_rest_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_table(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*/tables/*}' % client.transport._host, args[1])

def test_delete_table_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_table(metastore.DeleteTableRequest(), name='name_value')

def test_delete_table_rest_error():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.UpdateTableRequest, dict])
def test_update_table_rest(request_type):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'table': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}}
    request_init['table'] = {'hive_options': {'parameters': {}, 'table_type': 'table_type_value', 'storage_descriptor': {'location_uri': 'location_uri_value', 'input_format': 'input_format_value', 'output_format': 'output_format_value', 'serde_info': {'serialization_lib': 'serialization_lib_value'}}}, 'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}, 'type_': 1, 'etag': 'etag_value'}
    test_field = metastore.UpdateTableRequest.meta.fields['table']

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
    for (field, value) in request_init['table'].items():
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
                for i in range(0, len(request_init['table'][field])):
                    del request_init['table'][field][i][subfield]
            else:
                del request_init['table'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_table(request)
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_update_table_rest_required_fields(request_type=metastore.UpdateTableRequest):
    if False:
        return 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_table._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Table()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Table.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_table(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_table_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_table._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('table',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_table_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_update_table') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_update_table') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.UpdateTableRequest.pb(metastore.UpdateTableRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Table.to_json(metastore.Table())
        request = metastore.UpdateTableRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Table()
        client.update_table(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_table_rest_bad_request(transport: str='rest', request_type=metastore.UpdateTableRequest):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'table': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_table(request)

def test_update_table_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table()
        sample_request = {'table': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}}
        mock_args = dict(table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_table(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{table.name=projects/*/locations/*/catalogs/*/databases/*/tables/*}' % client.transport._host, args[1])

def test_update_table_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_table(metastore.UpdateTableRequest(), table=metastore.Table(hive_options=metastore.HiveTableOptions(parameters={'key_value': 'value_value'})), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_table_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.RenameTableRequest, dict])
def test_rename_table_rest(request_type):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.rename_table(request)
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_rename_table_rest_required_fields(request_type=metastore.RenameTableRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['new_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['newName'] = 'new_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'newName' in jsonified_request
    assert jsonified_request['newName'] == 'new_name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Table()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Table.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.rename_table(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_rename_table_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.rename_table._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'newName'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_rename_table_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_rename_table') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_rename_table') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.RenameTableRequest.pb(metastore.RenameTableRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Table.to_json(metastore.Table())
        request = metastore.RenameTableRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Table()
        client.rename_table(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_rename_table_rest_bad_request(transport: str='rest', request_type=metastore.RenameTableRequest):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.rename_table(request)

def test_rename_table_rest_flattened():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
        mock_args = dict(name='name_value', new_name='new_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.rename_table(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*/tables/*}:rename' % client.transport._host, args[1])

def test_rename_table_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.rename_table(metastore.RenameTableRequest(), name='name_value', new_name='new_name_value')

def test_rename_table_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.GetTableRequest, dict])
def test_get_table_rest(request_type):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table(name='name_value', type_=metastore.Table.Type.HIVE, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_table(request)
    assert isinstance(response, metastore.Table)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Table.Type.HIVE
    assert response.etag == 'etag_value'

def test_get_table_rest_required_fields(request_type=metastore.GetTableRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Table()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Table.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_table(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_table_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_table._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_table_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_get_table') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_get_table') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.GetTableRequest.pb(metastore.GetTableRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Table.to_json(metastore.Table())
        request = metastore.GetTableRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Table()
        client.get_table(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_table_rest_bad_request(transport: str='rest', request_type=metastore.GetTableRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_table(request)

def test_get_table_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Table()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/tables/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_table(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*/tables/*}' % client.transport._host, args[1])

def test_get_table_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_table(metastore.GetTableRequest(), name='name_value')

def test_get_table_rest_error():
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.ListTablesRequest, dict])
def test_list_tables_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListTablesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListTablesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tables(request)
    assert isinstance(response, pagers.ListTablesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tables_rest_required_fields(request_type=metastore.ListTablesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tables._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tables._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.ListTablesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.ListTablesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tables(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tables_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tables._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tables_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_list_tables') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_list_tables') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.ListTablesRequest.pb(metastore.ListTablesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.ListTablesResponse.to_json(metastore.ListTablesResponse())
        request = metastore.ListTablesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.ListTablesResponse()
        client.list_tables(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tables_rest_bad_request(transport: str='rest', request_type=metastore.ListTablesRequest):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tables(request)

def test_list_tables_rest_flattened():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListTablesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListTablesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tables(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*/catalogs/*/databases/*}/tables' % client.transport._host, args[1])

def test_list_tables_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tables(metastore.ListTablesRequest(), parent='parent_value')

def test_list_tables_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table(), metastore.Table()], next_page_token='abc'), metastore.ListTablesResponse(tables=[], next_page_token='def'), metastore.ListTablesResponse(tables=[metastore.Table()], next_page_token='ghi'), metastore.ListTablesResponse(tables=[metastore.Table(), metastore.Table()]))
        response = response + response
        response = tuple((metastore.ListTablesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        pager = client.list_tables(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Table) for i in results))
        pages = list(client.list_tables(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [metastore.CreateLockRequest, dict])
def test_create_lock_rest(request_type):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request_init['lock'] = {'table_id': 'table_id_value', 'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'type_': 1, 'state': 1}
    test_field = metastore.CreateLockRequest.meta.fields['lock']

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
    for (field, value) in request_init['lock'].items():
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
                for i in range(0, len(request_init['lock'][field])):
                    del request_init['lock'][field][i][subfield]
            else:
                del request_init['lock'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Lock(name='name_value', type_=metastore.Lock.Type.EXCLUSIVE, state=metastore.Lock.State.WAITING, table_id='table_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Lock.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_lock(request)
    assert isinstance(response, metastore.Lock)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Lock.Type.EXCLUSIVE
    assert response.state == metastore.Lock.State.WAITING

def test_create_lock_rest_required_fields(request_type=metastore.CreateLockRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_lock._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_lock._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Lock()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Lock.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_lock(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_lock_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_lock._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'lock'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_lock_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_create_lock') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_create_lock') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.CreateLockRequest.pb(metastore.CreateLockRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Lock.to_json(metastore.Lock())
        request = metastore.CreateLockRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Lock()
        client.create_lock(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_lock_rest_bad_request(transport: str='rest', request_type=metastore.CreateLockRequest):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_lock(request)

def test_create_lock_rest_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Lock()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        mock_args = dict(parent='parent_value', lock=metastore.Lock(table_id='table_id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Lock.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_lock(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*/catalogs/*/databases/*}/locks' % client.transport._host, args[1])

def test_create_lock_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_lock(metastore.CreateLockRequest(), parent='parent_value', lock=metastore.Lock(table_id='table_id_value'))

def test_create_lock_rest_error():
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.DeleteLockRequest, dict])
def test_delete_lock_rest(request_type):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/locks/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_lock(request)
    assert response is None

def test_delete_lock_rest_required_fields(request_type=metastore.DeleteLockRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_lock._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_lock._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_lock(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_lock_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_lock._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_lock_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_delete_lock') as pre:
        pre.assert_not_called()
        pb_message = metastore.DeleteLockRequest.pb(metastore.DeleteLockRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = metastore.DeleteLockRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_lock(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_lock_rest_bad_request(transport: str='rest', request_type=metastore.DeleteLockRequest):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/locks/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_lock(request)

def test_delete_lock_rest_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/locks/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_lock(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*/locks/*}' % client.transport._host, args[1])

def test_delete_lock_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_lock(metastore.DeleteLockRequest(), name='name_value')

def test_delete_lock_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.CheckLockRequest, dict])
def test_check_lock_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/locks/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Lock(name='name_value', type_=metastore.Lock.Type.EXCLUSIVE, state=metastore.Lock.State.WAITING, table_id='table_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Lock.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.check_lock(request)
    assert isinstance(response, metastore.Lock)
    assert response.name == 'name_value'
    assert response.type_ == metastore.Lock.Type.EXCLUSIVE
    assert response.state == metastore.Lock.State.WAITING

def test_check_lock_rest_required_fields(request_type=metastore.CheckLockRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).check_lock._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).check_lock._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.Lock()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.Lock.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.check_lock(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_check_lock_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.check_lock._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_check_lock_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_check_lock') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_check_lock') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.CheckLockRequest.pb(metastore.CheckLockRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.Lock.to_json(metastore.Lock())
        request = metastore.CheckLockRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.Lock()
        client.check_lock(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_check_lock_rest_bad_request(transport: str='rest', request_type=metastore.CheckLockRequest):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/locks/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.check_lock(request)

def test_check_lock_rest_flattened():
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.Lock()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4/locks/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.Lock.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.check_lock(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=projects/*/locations/*/catalogs/*/databases/*/locks/*}:check' % client.transport._host, args[1])

def test_check_lock_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.check_lock(metastore.CheckLockRequest(), name='name_value')

def test_check_lock_rest_error():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [metastore.ListLocksRequest, dict])
def test_list_locks_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListLocksResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListLocksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_locks(request)
    assert isinstance(response, pagers.ListLocksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_locks_rest_required_fields(request_type=metastore.ListLocksRequest):
    if False:
        print('Hello World!')
    transport_class = transports.MetastoreServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_locks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_locks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = metastore.ListLocksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = metastore.ListLocksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_locks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_locks_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_locks._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_locks_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.MetastoreServiceRestInterceptor())
    client = MetastoreServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'post_list_locks') as post, mock.patch.object(transports.MetastoreServiceRestInterceptor, 'pre_list_locks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = metastore.ListLocksRequest.pb(metastore.ListLocksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = metastore.ListLocksResponse.to_json(metastore.ListLocksResponse())
        request = metastore.ListLocksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = metastore.ListLocksResponse()
        client.list_locks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_locks_rest_bad_request(transport: str='rest', request_type=metastore.ListLocksRequest):
    if False:
        i = 10
        return i + 15
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_locks(request)

def test_list_locks_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = metastore.ListLocksResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = metastore.ListLocksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_locks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=projects/*/locations/*/catalogs/*/databases/*}/locks' % client.transport._host, args[1])

def test_list_locks_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_locks(metastore.ListLocksRequest(), parent='parent_value')

def test_list_locks_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock(), metastore.Lock()], next_page_token='abc'), metastore.ListLocksResponse(locks=[], next_page_token='def'), metastore.ListLocksResponse(locks=[metastore.Lock()], next_page_token='ghi'), metastore.ListLocksResponse(locks=[metastore.Lock(), metastore.Lock()]))
        response = response + response
        response = tuple((metastore.ListLocksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/databases/sample4'}
        pager = client.list_locks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, metastore.Lock) for i in results))
        pages = list(client.list_locks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.MetastoreServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.MetastoreServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MetastoreServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.MetastoreServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = MetastoreServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = MetastoreServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.MetastoreServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = MetastoreServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = MetastoreServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.MetastoreServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.MetastoreServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceGrpcAsyncIOTransport, transports.MetastoreServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = MetastoreServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.MetastoreServiceGrpcTransport)

def test_metastore_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.MetastoreServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_metastore_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.bigquery_biglake_v1alpha1.services.metastore_service.transports.MetastoreServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.MetastoreServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_catalog', 'delete_catalog', 'get_catalog', 'list_catalogs', 'create_database', 'delete_database', 'update_database', 'get_database', 'list_databases', 'create_table', 'delete_table', 'update_table', 'rename_table', 'get_table', 'list_tables', 'create_lock', 'delete_lock', 'check_lock', 'list_locks')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_metastore_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.bigquery_biglake_v1alpha1.services.metastore_service.transports.MetastoreServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.MetastoreServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_metastore_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.bigquery_biglake_v1alpha1.services.metastore_service.transports.MetastoreServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.MetastoreServiceTransport()
        adc.assert_called_once()

def test_metastore_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        MetastoreServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceGrpcAsyncIOTransport])
def test_metastore_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceGrpcAsyncIOTransport, transports.MetastoreServiceRestTransport])
def test_metastore_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.MetastoreServiceGrpcTransport, grpc_helpers), (transports.MetastoreServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_metastore_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('biglake.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), scopes=['1', '2'], default_host='biglake.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceGrpcAsyncIOTransport])
def test_metastore_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_metastore_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.MetastoreServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_metastore_service_host_no_port(transport_name):
    if False:
        return 10
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='biglake.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('biglake.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://biglake.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_metastore_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='biglake.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('biglake.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://biglake.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_metastore_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = MetastoreServiceClient(credentials=creds1, transport=transport_name)
    client2 = MetastoreServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_catalog._session
    session2 = client2.transport.create_catalog._session
    assert session1 != session2
    session1 = client1.transport.delete_catalog._session
    session2 = client2.transport.delete_catalog._session
    assert session1 != session2
    session1 = client1.transport.get_catalog._session
    session2 = client2.transport.get_catalog._session
    assert session1 != session2
    session1 = client1.transport.list_catalogs._session
    session2 = client2.transport.list_catalogs._session
    assert session1 != session2
    session1 = client1.transport.create_database._session
    session2 = client2.transport.create_database._session
    assert session1 != session2
    session1 = client1.transport.delete_database._session
    session2 = client2.transport.delete_database._session
    assert session1 != session2
    session1 = client1.transport.update_database._session
    session2 = client2.transport.update_database._session
    assert session1 != session2
    session1 = client1.transport.get_database._session
    session2 = client2.transport.get_database._session
    assert session1 != session2
    session1 = client1.transport.list_databases._session
    session2 = client2.transport.list_databases._session
    assert session1 != session2
    session1 = client1.transport.create_table._session
    session2 = client2.transport.create_table._session
    assert session1 != session2
    session1 = client1.transport.delete_table._session
    session2 = client2.transport.delete_table._session
    assert session1 != session2
    session1 = client1.transport.update_table._session
    session2 = client2.transport.update_table._session
    assert session1 != session2
    session1 = client1.transport.rename_table._session
    session2 = client2.transport.rename_table._session
    assert session1 != session2
    session1 = client1.transport.get_table._session
    session2 = client2.transport.get_table._session
    assert session1 != session2
    session1 = client1.transport.list_tables._session
    session2 = client2.transport.list_tables._session
    assert session1 != session2
    session1 = client1.transport.create_lock._session
    session2 = client2.transport.create_lock._session
    assert session1 != session2
    session1 = client1.transport.delete_lock._session
    session2 = client2.transport.delete_lock._session
    assert session1 != session2
    session1 = client1.transport.check_lock._session
    session2 = client2.transport.check_lock._session
    assert session1 != session2
    session1 = client1.transport.list_locks._session
    session2 = client2.transport.list_locks._session
    assert session1 != session2

def test_metastore_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.MetastoreServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_metastore_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.MetastoreServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceGrpcAsyncIOTransport])
def test_metastore_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.MetastoreServiceGrpcTransport, transports.MetastoreServiceGrpcAsyncIOTransport])
def test_metastore_service_transport_channel_mtls_with_adc(transport_class):
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

def test_catalog_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    catalog = 'whelk'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}'.format(project=project, location=location, catalog=catalog)
    actual = MetastoreServiceClient.catalog_path(project, location, catalog)
    assert expected == actual

def test_parse_catalog_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'catalog': 'nudibranch'}
    path = MetastoreServiceClient.catalog_path(**expected)
    actual = MetastoreServiceClient.parse_catalog_path(path)
    assert expected == actual

def test_database_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    catalog = 'winkle'
    database = 'nautilus'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/databases/{database}'.format(project=project, location=location, catalog=catalog, database=database)
    actual = MetastoreServiceClient.database_path(project, location, catalog, database)
    assert expected == actual

def test_parse_database_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone', 'catalog': 'squid', 'database': 'clam'}
    path = MetastoreServiceClient.database_path(**expected)
    actual = MetastoreServiceClient.parse_database_path(path)
    assert expected == actual

def test_lock_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    catalog = 'oyster'
    database = 'nudibranch'
    lock = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/databases/{database}/locks/{lock}'.format(project=project, location=location, catalog=catalog, database=database, lock=lock)
    actual = MetastoreServiceClient.lock_path(project, location, catalog, database, lock)
    assert expected == actual

def test_parse_lock_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel', 'location': 'winkle', 'catalog': 'nautilus', 'database': 'scallop', 'lock': 'abalone'}
    path = MetastoreServiceClient.lock_path(**expected)
    actual = MetastoreServiceClient.parse_lock_path(path)
    assert expected == actual

def test_table_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    catalog = 'whelk'
    database = 'octopus'
    table = 'oyster'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/databases/{database}/tables/{table}'.format(project=project, location=location, catalog=catalog, database=database, table=table)
    actual = MetastoreServiceClient.table_path(project, location, catalog, database, table)
    assert expected == actual

def test_parse_table_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'catalog': 'mussel', 'database': 'winkle', 'table': 'nautilus'}
    path = MetastoreServiceClient.table_path(**expected)
    actual = MetastoreServiceClient.parse_table_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = MetastoreServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'abalone'}
    path = MetastoreServiceClient.common_billing_account_path(**expected)
    actual = MetastoreServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = MetastoreServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = MetastoreServiceClient.common_folder_path(**expected)
    actual = MetastoreServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = MetastoreServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'octopus'}
    path = MetastoreServiceClient.common_organization_path(**expected)
    actual = MetastoreServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = MetastoreServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch'}
    path = MetastoreServiceClient.common_project_path(**expected)
    actual = MetastoreServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = MetastoreServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = MetastoreServiceClient.common_location_path(**expected)
    actual = MetastoreServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.MetastoreServiceTransport, '_prep_wrapped_messages') as prep:
        client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.MetastoreServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = MetastoreServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = MetastoreServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = MetastoreServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(MetastoreServiceClient, transports.MetastoreServiceGrpcTransport), (MetastoreServiceAsyncClient, transports.MetastoreServiceGrpcAsyncIOTransport)])
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