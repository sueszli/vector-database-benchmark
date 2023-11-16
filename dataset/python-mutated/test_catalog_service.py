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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import date_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.retail_v2.services.catalog_service import CatalogServiceAsyncClient, CatalogServiceClient, pagers, transports
from google.cloud.retail_v2.types import catalog
from google.cloud.retail_v2.types import catalog as gcr_catalog
from google.cloud.retail_v2.types import catalog_service, common, import_config

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CatalogServiceClient._get_default_mtls_endpoint(None) is None
    assert CatalogServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CatalogServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CatalogServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CatalogServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CatalogServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CatalogServiceClient, 'grpc'), (CatalogServiceAsyncClient, 'grpc_asyncio'), (CatalogServiceClient, 'rest')])
def test_catalog_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CatalogServiceGrpcTransport, 'grpc'), (transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CatalogServiceRestTransport, 'rest')])
def test_catalog_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CatalogServiceClient, 'grpc'), (CatalogServiceAsyncClient, 'grpc_asyncio'), (CatalogServiceClient, 'rest')])
def test_catalog_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

def test_catalog_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CatalogServiceClient.get_transport_class()
    available_transports = [transports.CatalogServiceGrpcTransport, transports.CatalogServiceRestTransport]
    assert transport in available_transports
    transport = CatalogServiceClient.get_transport_class('grpc')
    assert transport == transports.CatalogServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CatalogServiceClient, transports.CatalogServiceGrpcTransport, 'grpc'), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CatalogServiceClient, transports.CatalogServiceRestTransport, 'rest')])
@mock.patch.object(CatalogServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CatalogServiceClient))
@mock.patch.object(CatalogServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CatalogServiceAsyncClient))
def test_catalog_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(CatalogServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CatalogServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CatalogServiceClient, transports.CatalogServiceGrpcTransport, 'grpc', 'true'), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CatalogServiceClient, transports.CatalogServiceGrpcTransport, 'grpc', 'false'), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CatalogServiceClient, transports.CatalogServiceRestTransport, 'rest', 'true'), (CatalogServiceClient, transports.CatalogServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CatalogServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CatalogServiceClient))
@mock.patch.object(CatalogServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CatalogServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_catalog_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CatalogServiceClient, CatalogServiceAsyncClient])
@mock.patch.object(CatalogServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CatalogServiceClient))
@mock.patch.object(CatalogServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CatalogServiceAsyncClient))
def test_catalog_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CatalogServiceClient, transports.CatalogServiceGrpcTransport, 'grpc'), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CatalogServiceClient, transports.CatalogServiceRestTransport, 'rest')])
def test_catalog_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CatalogServiceClient, transports.CatalogServiceGrpcTransport, 'grpc', grpc_helpers), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CatalogServiceClient, transports.CatalogServiceRestTransport, 'rest', None)])
def test_catalog_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_catalog_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.retail_v2.services.catalog_service.transports.CatalogServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CatalogServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CatalogServiceClient, transports.CatalogServiceGrpcTransport, 'grpc', grpc_helpers), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_catalog_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [catalog_service.ListCatalogsRequest, dict])
def test_list_catalogs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = catalog_service.ListCatalogsResponse(next_page_token='next_page_token_value')
        response = client.list_catalogs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.ListCatalogsRequest()
    assert isinstance(response, pagers.ListCatalogsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_catalogs_empty_call():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        client.list_catalogs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.ListCatalogsRequest()

@pytest.mark.asyncio
async def test_list_catalogs_async(transport: str='grpc_asyncio', request_type=catalog_service.ListCatalogsRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog_service.ListCatalogsResponse(next_page_token='next_page_token_value'))
        response = await client.list_catalogs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.ListCatalogsRequest()
    assert isinstance(response, pagers.ListCatalogsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_catalogs_async_from_dict():
    await test_list_catalogs_async(request_type=dict)

def test_list_catalogs_field_headers():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.ListCatalogsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = catalog_service.ListCatalogsResponse()
        client.list_catalogs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_catalogs_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.ListCatalogsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog_service.ListCatalogsResponse())
        await client.list_catalogs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_catalogs_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = catalog_service.ListCatalogsResponse()
        client.list_catalogs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_catalogs_flattened_error():
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_catalogs(catalog_service.ListCatalogsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_catalogs_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.return_value = catalog_service.ListCatalogsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog_service.ListCatalogsResponse())
        response = await client.list_catalogs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_catalogs_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_catalogs(catalog_service.ListCatalogsRequest(), parent='parent_value')

def test_list_catalogs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.side_effect = (catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog(), catalog.Catalog()], next_page_token='abc'), catalog_service.ListCatalogsResponse(catalogs=[], next_page_token='def'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog()], next_page_token='ghi'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_catalogs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, catalog.Catalog) for i in results))

def test_list_catalogs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__') as call:
        call.side_effect = (catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog(), catalog.Catalog()], next_page_token='abc'), catalog_service.ListCatalogsResponse(catalogs=[], next_page_token='def'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog()], next_page_token='ghi'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog()]), RuntimeError)
        pages = list(client.list_catalogs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_catalogs_async_pager():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog(), catalog.Catalog()], next_page_token='abc'), catalog_service.ListCatalogsResponse(catalogs=[], next_page_token='def'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog()], next_page_token='ghi'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog()]), RuntimeError)
        async_pager = await client.list_catalogs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, catalog.Catalog) for i in responses))

@pytest.mark.asyncio
async def test_list_catalogs_async_pages():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_catalogs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog(), catalog.Catalog()], next_page_token='abc'), catalog_service.ListCatalogsResponse(catalogs=[], next_page_token='def'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog()], next_page_token='ghi'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_catalogs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [catalog_service.UpdateCatalogRequest, dict])
def test_update_catalog(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        call.return_value = gcr_catalog.Catalog(name='name_value', display_name='display_name_value')
        response = client.update_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateCatalogRequest()
    assert isinstance(response, gcr_catalog.Catalog)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_update_catalog_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        client.update_catalog()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateCatalogRequest()

@pytest.mark.asyncio
async def test_update_catalog_async(transport: str='grpc_asyncio', request_type=catalog_service.UpdateCatalogRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_catalog.Catalog(name='name_value', display_name='display_name_value'))
        response = await client.update_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateCatalogRequest()
    assert isinstance(response, gcr_catalog.Catalog)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_update_catalog_async_from_dict():
    await test_update_catalog_async(request_type=dict)

def test_update_catalog_field_headers():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.UpdateCatalogRequest()
    request.catalog.name = 'name_value'
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        call.return_value = gcr_catalog.Catalog()
        client.update_catalog(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'catalog.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_catalog_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.UpdateCatalogRequest()
    request.catalog.name = 'name_value'
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_catalog.Catalog())
        await client.update_catalog(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'catalog.name=name_value') in kw['metadata']

def test_update_catalog_flattened():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        call.return_value = gcr_catalog.Catalog()
        client.update_catalog(catalog=gcr_catalog.Catalog(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].catalog
        mock_val = gcr_catalog.Catalog(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_catalog_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_catalog(catalog_service.UpdateCatalogRequest(), catalog=gcr_catalog.Catalog(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_catalog_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_catalog), '__call__') as call:
        call.return_value = gcr_catalog.Catalog()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_catalog.Catalog())
        response = await client.update_catalog(catalog=gcr_catalog.Catalog(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].catalog
        mock_val = gcr_catalog.Catalog(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_catalog_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_catalog(catalog_service.UpdateCatalogRequest(), catalog=gcr_catalog.Catalog(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [catalog_service.SetDefaultBranchRequest, dict])
def test_set_default_branch(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        call.return_value = None
        response = client.set_default_branch(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.SetDefaultBranchRequest()
    assert response is None

def test_set_default_branch_empty_call():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        client.set_default_branch()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.SetDefaultBranchRequest()

@pytest.mark.asyncio
async def test_set_default_branch_async(transport: str='grpc_asyncio', request_type=catalog_service.SetDefaultBranchRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.set_default_branch(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.SetDefaultBranchRequest()
    assert response is None

@pytest.mark.asyncio
async def test_set_default_branch_async_from_dict():
    await test_set_default_branch_async(request_type=dict)

def test_set_default_branch_field_headers():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.SetDefaultBranchRequest()
    request.catalog = 'catalog_value'
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        call.return_value = None
        client.set_default_branch(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'catalog=catalog_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_default_branch_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.SetDefaultBranchRequest()
    request.catalog = 'catalog_value'
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.set_default_branch(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'catalog=catalog_value') in kw['metadata']

def test_set_default_branch_flattened():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        call.return_value = None
        client.set_default_branch(catalog='catalog_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].catalog
        mock_val = 'catalog_value'
        assert arg == mock_val

def test_set_default_branch_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_default_branch(catalog_service.SetDefaultBranchRequest(), catalog='catalog_value')

@pytest.mark.asyncio
async def test_set_default_branch_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_default_branch), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.set_default_branch(catalog='catalog_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].catalog
        mock_val = 'catalog_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_default_branch_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_default_branch(catalog_service.SetDefaultBranchRequest(), catalog='catalog_value')

@pytest.mark.parametrize('request_type', [catalog_service.GetDefaultBranchRequest, dict])
def test_get_default_branch(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        call.return_value = catalog_service.GetDefaultBranchResponse(branch='branch_value', note='note_value')
        response = client.get_default_branch(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetDefaultBranchRequest()
    assert isinstance(response, catalog_service.GetDefaultBranchResponse)
    assert response.branch == 'branch_value'
    assert response.note == 'note_value'

def test_get_default_branch_empty_call():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        client.get_default_branch()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetDefaultBranchRequest()

@pytest.mark.asyncio
async def test_get_default_branch_async(transport: str='grpc_asyncio', request_type=catalog_service.GetDefaultBranchRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog_service.GetDefaultBranchResponse(branch='branch_value', note='note_value'))
        response = await client.get_default_branch(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetDefaultBranchRequest()
    assert isinstance(response, catalog_service.GetDefaultBranchResponse)
    assert response.branch == 'branch_value'
    assert response.note == 'note_value'

@pytest.mark.asyncio
async def test_get_default_branch_async_from_dict():
    await test_get_default_branch_async(request_type=dict)

def test_get_default_branch_field_headers():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.GetDefaultBranchRequest()
    request.catalog = 'catalog_value'
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        call.return_value = catalog_service.GetDefaultBranchResponse()
        client.get_default_branch(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'catalog=catalog_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_default_branch_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.GetDefaultBranchRequest()
    request.catalog = 'catalog_value'
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog_service.GetDefaultBranchResponse())
        await client.get_default_branch(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'catalog=catalog_value') in kw['metadata']

def test_get_default_branch_flattened():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        call.return_value = catalog_service.GetDefaultBranchResponse()
        client.get_default_branch(catalog='catalog_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].catalog
        mock_val = 'catalog_value'
        assert arg == mock_val

def test_get_default_branch_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_default_branch(catalog_service.GetDefaultBranchRequest(), catalog='catalog_value')

@pytest.mark.asyncio
async def test_get_default_branch_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_default_branch), '__call__') as call:
        call.return_value = catalog_service.GetDefaultBranchResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog_service.GetDefaultBranchResponse())
        response = await client.get_default_branch(catalog='catalog_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].catalog
        mock_val = 'catalog_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_default_branch_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_default_branch(catalog_service.GetDefaultBranchRequest(), catalog='catalog_value')

@pytest.mark.parametrize('request_type', [catalog_service.GetCompletionConfigRequest, dict])
def test_get_completion_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig(name='name_value', matching_order='matching_order_value', max_suggestions=1632, min_prefix_length=1810, auto_learning=True, last_suggestions_import_operation='last_suggestions_import_operation_value', last_denylist_import_operation='last_denylist_import_operation_value', last_allowlist_import_operation='last_allowlist_import_operation_value')
        response = client.get_completion_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetCompletionConfigRequest()
    assert isinstance(response, catalog.CompletionConfig)
    assert response.name == 'name_value'
    assert response.matching_order == 'matching_order_value'
    assert response.max_suggestions == 1632
    assert response.min_prefix_length == 1810
    assert response.auto_learning is True
    assert response.last_suggestions_import_operation == 'last_suggestions_import_operation_value'
    assert response.last_denylist_import_operation == 'last_denylist_import_operation_value'
    assert response.last_allowlist_import_operation == 'last_allowlist_import_operation_value'

def test_get_completion_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        client.get_completion_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetCompletionConfigRequest()

@pytest.mark.asyncio
async def test_get_completion_config_async(transport: str='grpc_asyncio', request_type=catalog_service.GetCompletionConfigRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.CompletionConfig(name='name_value', matching_order='matching_order_value', max_suggestions=1632, min_prefix_length=1810, auto_learning=True, last_suggestions_import_operation='last_suggestions_import_operation_value', last_denylist_import_operation='last_denylist_import_operation_value', last_allowlist_import_operation='last_allowlist_import_operation_value'))
        response = await client.get_completion_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetCompletionConfigRequest()
    assert isinstance(response, catalog.CompletionConfig)
    assert response.name == 'name_value'
    assert response.matching_order == 'matching_order_value'
    assert response.max_suggestions == 1632
    assert response.min_prefix_length == 1810
    assert response.auto_learning is True
    assert response.last_suggestions_import_operation == 'last_suggestions_import_operation_value'
    assert response.last_denylist_import_operation == 'last_denylist_import_operation_value'
    assert response.last_allowlist_import_operation == 'last_allowlist_import_operation_value'

@pytest.mark.asyncio
async def test_get_completion_config_async_from_dict():
    await test_get_completion_config_async(request_type=dict)

def test_get_completion_config_field_headers():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.GetCompletionConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig()
        client.get_completion_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_completion_config_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.GetCompletionConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.CompletionConfig())
        await client.get_completion_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_completion_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig()
        client.get_completion_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_completion_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_completion_config(catalog_service.GetCompletionConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_completion_config_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.CompletionConfig())
        response = await client.get_completion_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_completion_config_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_completion_config(catalog_service.GetCompletionConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [catalog_service.UpdateCompletionConfigRequest, dict])
def test_update_completion_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig(name='name_value', matching_order='matching_order_value', max_suggestions=1632, min_prefix_length=1810, auto_learning=True, last_suggestions_import_operation='last_suggestions_import_operation_value', last_denylist_import_operation='last_denylist_import_operation_value', last_allowlist_import_operation='last_allowlist_import_operation_value')
        response = client.update_completion_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateCompletionConfigRequest()
    assert isinstance(response, catalog.CompletionConfig)
    assert response.name == 'name_value'
    assert response.matching_order == 'matching_order_value'
    assert response.max_suggestions == 1632
    assert response.min_prefix_length == 1810
    assert response.auto_learning is True
    assert response.last_suggestions_import_operation == 'last_suggestions_import_operation_value'
    assert response.last_denylist_import_operation == 'last_denylist_import_operation_value'
    assert response.last_allowlist_import_operation == 'last_allowlist_import_operation_value'

def test_update_completion_config_empty_call():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        client.update_completion_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateCompletionConfigRequest()

@pytest.mark.asyncio
async def test_update_completion_config_async(transport: str='grpc_asyncio', request_type=catalog_service.UpdateCompletionConfigRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.CompletionConfig(name='name_value', matching_order='matching_order_value', max_suggestions=1632, min_prefix_length=1810, auto_learning=True, last_suggestions_import_operation='last_suggestions_import_operation_value', last_denylist_import_operation='last_denylist_import_operation_value', last_allowlist_import_operation='last_allowlist_import_operation_value'))
        response = await client.update_completion_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateCompletionConfigRequest()
    assert isinstance(response, catalog.CompletionConfig)
    assert response.name == 'name_value'
    assert response.matching_order == 'matching_order_value'
    assert response.max_suggestions == 1632
    assert response.min_prefix_length == 1810
    assert response.auto_learning is True
    assert response.last_suggestions_import_operation == 'last_suggestions_import_operation_value'
    assert response.last_denylist_import_operation == 'last_denylist_import_operation_value'
    assert response.last_allowlist_import_operation == 'last_allowlist_import_operation_value'

@pytest.mark.asyncio
async def test_update_completion_config_async_from_dict():
    await test_update_completion_config_async(request_type=dict)

def test_update_completion_config_field_headers():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.UpdateCompletionConfigRequest()
    request.completion_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig()
        client.update_completion_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'completion_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_completion_config_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.UpdateCompletionConfigRequest()
    request.completion_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.CompletionConfig())
        await client.update_completion_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'completion_config.name=name_value') in kw['metadata']

def test_update_completion_config_flattened():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig()
        client.update_completion_config(completion_config=catalog.CompletionConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].completion_config
        mock_val = catalog.CompletionConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_completion_config_flattened_error():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_completion_config(catalog_service.UpdateCompletionConfigRequest(), completion_config=catalog.CompletionConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_completion_config_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_completion_config), '__call__') as call:
        call.return_value = catalog.CompletionConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.CompletionConfig())
        response = await client.update_completion_config(completion_config=catalog.CompletionConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].completion_config
        mock_val = catalog.CompletionConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_completion_config_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_completion_config(catalog_service.UpdateCompletionConfigRequest(), completion_config=catalog.CompletionConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [catalog_service.GetAttributesConfigRequest, dict])
def test_get_attributes_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response = client.get_attributes_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetAttributesConfigRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_get_attributes_config_empty_call():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        client.get_attributes_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetAttributesConfigRequest()

@pytest.mark.asyncio
async def test_get_attributes_config_async(transport: str='grpc_asyncio', request_type=catalog_service.GetAttributesConfigRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG))
        response = await client.get_attributes_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.GetAttributesConfigRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

@pytest.mark.asyncio
async def test_get_attributes_config_async_from_dict():
    await test_get_attributes_config_async(request_type=dict)

def test_get_attributes_config_field_headers():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.GetAttributesConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.get_attributes_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_attributes_config_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.GetAttributesConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        await client.get_attributes_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_attributes_config_flattened():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.get_attributes_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_attributes_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_attributes_config(catalog_service.GetAttributesConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_attributes_config_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        response = await client.get_attributes_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_attributes_config_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_attributes_config(catalog_service.GetAttributesConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [catalog_service.UpdateAttributesConfigRequest, dict])
def test_update_attributes_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response = client.update_attributes_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateAttributesConfigRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_update_attributes_config_empty_call():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        client.update_attributes_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateAttributesConfigRequest()

@pytest.mark.asyncio
async def test_update_attributes_config_async(transport: str='grpc_asyncio', request_type=catalog_service.UpdateAttributesConfigRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG))
        response = await client.update_attributes_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.UpdateAttributesConfigRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

@pytest.mark.asyncio
async def test_update_attributes_config_async_from_dict():
    await test_update_attributes_config_async(request_type=dict)

def test_update_attributes_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.UpdateAttributesConfigRequest()
    request.attributes_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.update_attributes_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_attributes_config_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.UpdateAttributesConfigRequest()
    request.attributes_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        await client.update_attributes_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config.name=name_value') in kw['metadata']

def test_update_attributes_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.update_attributes_config(attributes_config=catalog.AttributesConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].attributes_config
        mock_val = catalog.AttributesConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_attributes_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_attributes_config(catalog_service.UpdateAttributesConfigRequest(), attributes_config=catalog.AttributesConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_attributes_config_flattened_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_attributes_config), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        response = await client.update_attributes_config(attributes_config=catalog.AttributesConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].attributes_config
        mock_val = catalog.AttributesConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_attributes_config_flattened_error_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_attributes_config(catalog_service.UpdateAttributesConfigRequest(), attributes_config=catalog.AttributesConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [catalog_service.AddCatalogAttributeRequest, dict])
def test_add_catalog_attribute(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_catalog_attribute), '__call__') as call:
        call.return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response = client.add_catalog_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.AddCatalogAttributeRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_add_catalog_attribute_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.add_catalog_attribute), '__call__') as call:
        client.add_catalog_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.AddCatalogAttributeRequest()

@pytest.mark.asyncio
async def test_add_catalog_attribute_async(transport: str='grpc_asyncio', request_type=catalog_service.AddCatalogAttributeRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_catalog_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG))
        response = await client.add_catalog_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.AddCatalogAttributeRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

@pytest.mark.asyncio
async def test_add_catalog_attribute_async_from_dict():
    await test_add_catalog_attribute_async(request_type=dict)

def test_add_catalog_attribute_field_headers():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.AddCatalogAttributeRequest()
    request.attributes_config = 'attributes_config_value'
    with mock.patch.object(type(client.transport.add_catalog_attribute), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.add_catalog_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config=attributes_config_value') in kw['metadata']

@pytest.mark.asyncio
async def test_add_catalog_attribute_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.AddCatalogAttributeRequest()
    request.attributes_config = 'attributes_config_value'
    with mock.patch.object(type(client.transport.add_catalog_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        await client.add_catalog_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config=attributes_config_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [catalog_service.RemoveCatalogAttributeRequest, dict])
def test_remove_catalog_attribute(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_catalog_attribute), '__call__') as call:
        call.return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response = client.remove_catalog_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.RemoveCatalogAttributeRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_remove_catalog_attribute_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.remove_catalog_attribute), '__call__') as call:
        client.remove_catalog_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.RemoveCatalogAttributeRequest()

@pytest.mark.asyncio
async def test_remove_catalog_attribute_async(transport: str='grpc_asyncio', request_type=catalog_service.RemoveCatalogAttributeRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_catalog_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG))
        response = await client.remove_catalog_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.RemoveCatalogAttributeRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

@pytest.mark.asyncio
async def test_remove_catalog_attribute_async_from_dict():
    await test_remove_catalog_attribute_async(request_type=dict)

def test_remove_catalog_attribute_field_headers():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.RemoveCatalogAttributeRequest()
    request.attributes_config = 'attributes_config_value'
    with mock.patch.object(type(client.transport.remove_catalog_attribute), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.remove_catalog_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config=attributes_config_value') in kw['metadata']

@pytest.mark.asyncio
async def test_remove_catalog_attribute_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.RemoveCatalogAttributeRequest()
    request.attributes_config = 'attributes_config_value'
    with mock.patch.object(type(client.transport.remove_catalog_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        await client.remove_catalog_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config=attributes_config_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [catalog_service.ReplaceCatalogAttributeRequest, dict])
def test_replace_catalog_attribute(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.replace_catalog_attribute), '__call__') as call:
        call.return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response = client.replace_catalog_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.ReplaceCatalogAttributeRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_replace_catalog_attribute_empty_call():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.replace_catalog_attribute), '__call__') as call:
        client.replace_catalog_attribute()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.ReplaceCatalogAttributeRequest()

@pytest.mark.asyncio
async def test_replace_catalog_attribute_async(transport: str='grpc_asyncio', request_type=catalog_service.ReplaceCatalogAttributeRequest):
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.replace_catalog_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG))
        response = await client.replace_catalog_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == catalog_service.ReplaceCatalogAttributeRequest()
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

@pytest.mark.asyncio
async def test_replace_catalog_attribute_async_from_dict():
    await test_replace_catalog_attribute_async(request_type=dict)

def test_replace_catalog_attribute_field_headers():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.ReplaceCatalogAttributeRequest()
    request.attributes_config = 'attributes_config_value'
    with mock.patch.object(type(client.transport.replace_catalog_attribute), '__call__') as call:
        call.return_value = catalog.AttributesConfig()
        client.replace_catalog_attribute(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config=attributes_config_value') in kw['metadata']

@pytest.mark.asyncio
async def test_replace_catalog_attribute_field_headers_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = catalog_service.ReplaceCatalogAttributeRequest()
    request.attributes_config = 'attributes_config_value'
    with mock.patch.object(type(client.transport.replace_catalog_attribute), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(catalog.AttributesConfig())
        await client.replace_catalog_attribute(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attributes_config=attributes_config_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [catalog_service.ListCatalogsRequest, dict])
def test_list_catalogs_rest(request_type):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog_service.ListCatalogsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog_service.ListCatalogsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_catalogs(request)
    assert isinstance(response, pagers.ListCatalogsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_catalogs_rest_required_fields(request_type=catalog_service.ListCatalogsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CatalogServiceRestTransport
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
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog_service.ListCatalogsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog_service.ListCatalogsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_catalogs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_catalogs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_catalogs._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_catalogs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_list_catalogs') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_list_catalogs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.ListCatalogsRequest.pb(catalog_service.ListCatalogsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog_service.ListCatalogsResponse.to_json(catalog_service.ListCatalogsResponse())
        request = catalog_service.ListCatalogsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog_service.ListCatalogsResponse()
        client.list_catalogs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_catalogs_rest_bad_request(transport: str='rest', request_type=catalog_service.ListCatalogsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog_service.ListCatalogsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog_service.ListCatalogsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_catalogs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/catalogs' % client.transport._host, args[1])

def test_list_catalogs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_catalogs(catalog_service.ListCatalogsRequest(), parent='parent_value')

def test_list_catalogs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog(), catalog.Catalog()], next_page_token='abc'), catalog_service.ListCatalogsResponse(catalogs=[], next_page_token='def'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog()], next_page_token='ghi'), catalog_service.ListCatalogsResponse(catalogs=[catalog.Catalog(), catalog.Catalog()]))
        response = response + response
        response = tuple((catalog_service.ListCatalogsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_catalogs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, catalog.Catalog) for i in results))
        pages = list(client.list_catalogs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [catalog_service.UpdateCatalogRequest, dict])
def test_update_catalog_rest(request_type):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'catalog': {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}}
    request_init['catalog'] = {'name': 'projects/sample1/locations/sample2/catalogs/sample3', 'display_name': 'display_name_value', 'product_level_config': {'ingestion_product_type': 'ingestion_product_type_value', 'merchant_center_product_id_field': 'merchant_center_product_id_field_value'}}
    test_field = catalog_service.UpdateCatalogRequest.meta.fields['catalog']

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
        return_value = gcr_catalog.Catalog(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_catalog.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_catalog(request)
    assert isinstance(response, gcr_catalog.Catalog)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_update_catalog_rest_required_fields(request_type=catalog_service.UpdateCatalogRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_catalog._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_catalog._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcr_catalog.Catalog()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcr_catalog.Catalog.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_catalog(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_catalog_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_catalog._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('catalog',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_catalog_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_update_catalog') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_update_catalog') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.UpdateCatalogRequest.pb(catalog_service.UpdateCatalogRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcr_catalog.Catalog.to_json(gcr_catalog.Catalog())
        request = catalog_service.UpdateCatalogRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcr_catalog.Catalog()
        client.update_catalog(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_catalog_rest_bad_request(transport: str='rest', request_type=catalog_service.UpdateCatalogRequest):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'catalog': {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_catalog(request)

def test_update_catalog_rest_flattened():
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_catalog.Catalog()
        sample_request = {'catalog': {'name': 'projects/sample1/locations/sample2/catalogs/sample3'}}
        mock_args = dict(catalog=gcr_catalog.Catalog(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_catalog.Catalog.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_catalog(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{catalog.name=projects/*/locations/*/catalogs/*}' % client.transport._host, args[1])

def test_update_catalog_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_catalog(catalog_service.UpdateCatalogRequest(), catalog=gcr_catalog.Catalog(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_catalog_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.SetDefaultBranchRequest, dict])
def test_set_default_branch_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'catalog': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_default_branch(request)
    assert response is None

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_default_branch_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_set_default_branch') as pre:
        pre.assert_not_called()
        pb_message = catalog_service.SetDefaultBranchRequest.pb(catalog_service.SetDefaultBranchRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = catalog_service.SetDefaultBranchRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.set_default_branch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_set_default_branch_rest_bad_request(transport: str='rest', request_type=catalog_service.SetDefaultBranchRequest):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'catalog': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_default_branch(request)

def test_set_default_branch_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'catalog': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(catalog='catalog_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_default_branch(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{catalog=projects/*/locations/*/catalogs/*}:setDefaultBranch' % client.transport._host, args[1])

def test_set_default_branch_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_default_branch(catalog_service.SetDefaultBranchRequest(), catalog='catalog_value')

def test_set_default_branch_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.GetDefaultBranchRequest, dict])
def test_get_default_branch_rest(request_type):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'catalog': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog_service.GetDefaultBranchResponse(branch='branch_value', note='note_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog_service.GetDefaultBranchResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_default_branch(request)
    assert isinstance(response, catalog_service.GetDefaultBranchResponse)
    assert response.branch == 'branch_value'
    assert response.note == 'note_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_default_branch_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_get_default_branch') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_get_default_branch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.GetDefaultBranchRequest.pb(catalog_service.GetDefaultBranchRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog_service.GetDefaultBranchResponse.to_json(catalog_service.GetDefaultBranchResponse())
        request = catalog_service.GetDefaultBranchRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog_service.GetDefaultBranchResponse()
        client.get_default_branch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_default_branch_rest_bad_request(transport: str='rest', request_type=catalog_service.GetDefaultBranchRequest):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'catalog': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_default_branch(request)

def test_get_default_branch_rest_flattened():
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog_service.GetDefaultBranchResponse()
        sample_request = {'catalog': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(catalog='catalog_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog_service.GetDefaultBranchResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_default_branch(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{catalog=projects/*/locations/*/catalogs/*}:getDefaultBranch' % client.transport._host, args[1])

def test_get_default_branch_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_default_branch(catalog_service.GetDefaultBranchRequest(), catalog='catalog_value')

def test_get_default_branch_rest_error():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.GetCompletionConfigRequest, dict])
def test_get_completion_config_rest(request_type):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.CompletionConfig(name='name_value', matching_order='matching_order_value', max_suggestions=1632, min_prefix_length=1810, auto_learning=True, last_suggestions_import_operation='last_suggestions_import_operation_value', last_denylist_import_operation='last_denylist_import_operation_value', last_allowlist_import_operation='last_allowlist_import_operation_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.CompletionConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_completion_config(request)
    assert isinstance(response, catalog.CompletionConfig)
    assert response.name == 'name_value'
    assert response.matching_order == 'matching_order_value'
    assert response.max_suggestions == 1632
    assert response.min_prefix_length == 1810
    assert response.auto_learning is True
    assert response.last_suggestions_import_operation == 'last_suggestions_import_operation_value'
    assert response.last_denylist_import_operation == 'last_denylist_import_operation_value'
    assert response.last_allowlist_import_operation == 'last_allowlist_import_operation_value'

def test_get_completion_config_rest_required_fields(request_type=catalog_service.GetCompletionConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_completion_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_completion_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.CompletionConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.CompletionConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_completion_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_completion_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_completion_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_completion_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_get_completion_config') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_get_completion_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.GetCompletionConfigRequest.pb(catalog_service.GetCompletionConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.CompletionConfig.to_json(catalog.CompletionConfig())
        request = catalog_service.GetCompletionConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.CompletionConfig()
        client.get_completion_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_completion_config_rest_bad_request(transport: str='rest', request_type=catalog_service.GetCompletionConfigRequest):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_completion_config(request)

def test_get_completion_config_rest_flattened():
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.CompletionConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.CompletionConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_completion_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/catalogs/*/completionConfig}' % client.transport._host, args[1])

def test_get_completion_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_completion_config(catalog_service.GetCompletionConfigRequest(), name='name_value')

def test_get_completion_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.UpdateCompletionConfigRequest, dict])
def test_update_completion_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'completion_config': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig'}}
    request_init['completion_config'] = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig', 'matching_order': 'matching_order_value', 'max_suggestions': 1632, 'min_prefix_length': 1810, 'auto_learning': True, 'suggestions_input_config': {'big_query_source': {'partition_date': {'year': 433, 'month': 550, 'day': 318}, 'project_id': 'project_id_value', 'dataset_id': 'dataset_id_value', 'table_id': 'table_id_value', 'gcs_staging_dir': 'gcs_staging_dir_value', 'data_schema': 'data_schema_value'}}, 'last_suggestions_import_operation': 'last_suggestions_import_operation_value', 'denylist_input_config': {}, 'last_denylist_import_operation': 'last_denylist_import_operation_value', 'allowlist_input_config': {}, 'last_allowlist_import_operation': 'last_allowlist_import_operation_value'}
    test_field = catalog_service.UpdateCompletionConfigRequest.meta.fields['completion_config']

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
    for (field, value) in request_init['completion_config'].items():
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
                for i in range(0, len(request_init['completion_config'][field])):
                    del request_init['completion_config'][field][i][subfield]
            else:
                del request_init['completion_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.CompletionConfig(name='name_value', matching_order='matching_order_value', max_suggestions=1632, min_prefix_length=1810, auto_learning=True, last_suggestions_import_operation='last_suggestions_import_operation_value', last_denylist_import_operation='last_denylist_import_operation_value', last_allowlist_import_operation='last_allowlist_import_operation_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.CompletionConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_completion_config(request)
    assert isinstance(response, catalog.CompletionConfig)
    assert response.name == 'name_value'
    assert response.matching_order == 'matching_order_value'
    assert response.max_suggestions == 1632
    assert response.min_prefix_length == 1810
    assert response.auto_learning is True
    assert response.last_suggestions_import_operation == 'last_suggestions_import_operation_value'
    assert response.last_denylist_import_operation == 'last_denylist_import_operation_value'
    assert response.last_allowlist_import_operation == 'last_allowlist_import_operation_value'

def test_update_completion_config_rest_required_fields(request_type=catalog_service.UpdateCompletionConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_completion_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_completion_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.CompletionConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.CompletionConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_completion_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_completion_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_completion_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('completionConfig',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_completion_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_update_completion_config') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_update_completion_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.UpdateCompletionConfigRequest.pb(catalog_service.UpdateCompletionConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.CompletionConfig.to_json(catalog.CompletionConfig())
        request = catalog_service.UpdateCompletionConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.CompletionConfig()
        client.update_completion_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_completion_config_rest_bad_request(transport: str='rest', request_type=catalog_service.UpdateCompletionConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'completion_config': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_completion_config(request)

def test_update_completion_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.CompletionConfig()
        sample_request = {'completion_config': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/completionConfig'}}
        mock_args = dict(completion_config=catalog.CompletionConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.CompletionConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_completion_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{completion_config.name=projects/*/locations/*/catalogs/*/completionConfig}' % client.transport._host, args[1])

def test_update_completion_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_completion_config(catalog_service.UpdateCompletionConfigRequest(), completion_config=catalog.CompletionConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_completion_config_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.GetAttributesConfigRequest, dict])
def test_get_attributes_config_rest(request_type):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_attributes_config(request)
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_get_attributes_config_rest_required_fields(request_type=catalog_service.GetAttributesConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_attributes_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_attributes_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.AttributesConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.AttributesConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_attributes_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_attributes_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_attributes_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_attributes_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_get_attributes_config') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_get_attributes_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.GetAttributesConfigRequest.pb(catalog_service.GetAttributesConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.AttributesConfig.to_json(catalog.AttributesConfig())
        request = catalog_service.GetAttributesConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.AttributesConfig()
        client.get_attributes_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_attributes_config_rest_bad_request(transport: str='rest', request_type=catalog_service.GetAttributesConfigRequest):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_attributes_config(request)

def test_get_attributes_config_rest_flattened():
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_attributes_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/catalogs/*/attributesConfig}' % client.transport._host, args[1])

def test_get_attributes_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_attributes_config(catalog_service.GetAttributesConfigRequest(), name='name_value')

def test_get_attributes_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.UpdateAttributesConfigRequest, dict])
def test_update_attributes_config_rest(request_type):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'attributes_config': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}}
    request_init['attributes_config'] = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig', 'catalog_attributes': {}, 'attribute_config_level': 1}
    test_field = catalog_service.UpdateAttributesConfigRequest.meta.fields['attributes_config']

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
    for (field, value) in request_init['attributes_config'].items():
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
                for i in range(0, len(request_init['attributes_config'][field])):
                    del request_init['attributes_config'][field][i][subfield]
            else:
                del request_init['attributes_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_attributes_config(request)
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_update_attributes_config_rest_required_fields(request_type=catalog_service.UpdateAttributesConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_attributes_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_attributes_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.AttributesConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.AttributesConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_attributes_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_attributes_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_attributes_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('attributesConfig',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_attributes_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_update_attributes_config') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_update_attributes_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.UpdateAttributesConfigRequest.pb(catalog_service.UpdateAttributesConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.AttributesConfig.to_json(catalog.AttributesConfig())
        request = catalog_service.UpdateAttributesConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.AttributesConfig()
        client.update_attributes_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_attributes_config_rest_bad_request(transport: str='rest', request_type=catalog_service.UpdateAttributesConfigRequest):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'attributes_config': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_attributes_config(request)

def test_update_attributes_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig()
        sample_request = {'attributes_config': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}}
        mock_args = dict(attributes_config=catalog.AttributesConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_attributes_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{attributes_config.name=projects/*/locations/*/catalogs/*/attributesConfig}' % client.transport._host, args[1])

def test_update_attributes_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_attributes_config(catalog_service.UpdateAttributesConfigRequest(), attributes_config=catalog.AttributesConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_attributes_config_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.AddCatalogAttributeRequest, dict])
def test_add_catalog_attribute_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'attributes_config': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_catalog_attribute(request)
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_add_catalog_attribute_rest_required_fields(request_type=catalog_service.AddCatalogAttributeRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request_init['attributes_config'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_catalog_attribute._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['attributesConfig'] = 'attributes_config_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_catalog_attribute._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'attributesConfig' in jsonified_request
    assert jsonified_request['attributesConfig'] == 'attributes_config_value'
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.AttributesConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.AttributesConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.add_catalog_attribute(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_catalog_attribute_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_catalog_attribute._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('attributesConfig', 'catalogAttribute'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_catalog_attribute_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_add_catalog_attribute') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_add_catalog_attribute') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.AddCatalogAttributeRequest.pb(catalog_service.AddCatalogAttributeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.AttributesConfig.to_json(catalog.AttributesConfig())
        request = catalog_service.AddCatalogAttributeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.AttributesConfig()
        client.add_catalog_attribute(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_catalog_attribute_rest_bad_request(transport: str='rest', request_type=catalog_service.AddCatalogAttributeRequest):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'attributes_config': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_catalog_attribute(request)

def test_add_catalog_attribute_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.RemoveCatalogAttributeRequest, dict])
def test_remove_catalog_attribute_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'attributes_config': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_catalog_attribute(request)
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_remove_catalog_attribute_rest_required_fields(request_type=catalog_service.RemoveCatalogAttributeRequest):
    if False:
        return 10
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request_init['attributes_config'] = ''
    request_init['key'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_catalog_attribute._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['attributesConfig'] = 'attributes_config_value'
    jsonified_request['key'] = 'key_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_catalog_attribute._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'attributesConfig' in jsonified_request
    assert jsonified_request['attributesConfig'] == 'attributes_config_value'
    assert 'key' in jsonified_request
    assert jsonified_request['key'] == 'key_value'
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.AttributesConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.AttributesConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.remove_catalog_attribute(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_catalog_attribute_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_catalog_attribute._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('attributesConfig', 'key'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_catalog_attribute_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_remove_catalog_attribute') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_remove_catalog_attribute') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.RemoveCatalogAttributeRequest.pb(catalog_service.RemoveCatalogAttributeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.AttributesConfig.to_json(catalog.AttributesConfig())
        request = catalog_service.RemoveCatalogAttributeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.AttributesConfig()
        client.remove_catalog_attribute(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_catalog_attribute_rest_bad_request(transport: str='rest', request_type=catalog_service.RemoveCatalogAttributeRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'attributes_config': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_catalog_attribute(request)

def test_remove_catalog_attribute_rest_error():
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [catalog_service.ReplaceCatalogAttributeRequest, dict])
def test_replace_catalog_attribute_rest(request_type):
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'attributes_config': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = catalog.AttributesConfig(name='name_value', attribute_config_level=common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG)
        response_value = Response()
        response_value.status_code = 200
        return_value = catalog.AttributesConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.replace_catalog_attribute(request)
    assert isinstance(response, catalog.AttributesConfig)
    assert response.name == 'name_value'
    assert response.attribute_config_level == common.AttributeConfigLevel.PRODUCT_LEVEL_ATTRIBUTE_CONFIG

def test_replace_catalog_attribute_rest_required_fields(request_type=catalog_service.ReplaceCatalogAttributeRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CatalogServiceRestTransport
    request_init = {}
    request_init['attributes_config'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).replace_catalog_attribute._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['attributesConfig'] = 'attributes_config_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).replace_catalog_attribute._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'attributesConfig' in jsonified_request
    assert jsonified_request['attributesConfig'] == 'attributes_config_value'
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = catalog.AttributesConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = catalog.AttributesConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.replace_catalog_attribute(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_replace_catalog_attribute_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.replace_catalog_attribute._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('attributesConfig', 'catalogAttribute'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_replace_catalog_attribute_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CatalogServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CatalogServiceRestInterceptor())
    client = CatalogServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CatalogServiceRestInterceptor, 'post_replace_catalog_attribute') as post, mock.patch.object(transports.CatalogServiceRestInterceptor, 'pre_replace_catalog_attribute') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = catalog_service.ReplaceCatalogAttributeRequest.pb(catalog_service.ReplaceCatalogAttributeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = catalog.AttributesConfig.to_json(catalog.AttributesConfig())
        request = catalog_service.ReplaceCatalogAttributeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = catalog.AttributesConfig()
        client.replace_catalog_attribute(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_replace_catalog_attribute_rest_bad_request(transport: str='rest', request_type=catalog_service.ReplaceCatalogAttributeRequest):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'attributes_config': 'projects/sample1/locations/sample2/catalogs/sample3/attributesConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.replace_catalog_attribute(request)

def test_replace_catalog_attribute_rest_error():
    if False:
        while True:
            i = 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CatalogServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CatalogServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CatalogServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CatalogServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CatalogServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CatalogServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CatalogServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CatalogServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.CatalogServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CatalogServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.CatalogServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CatalogServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CatalogServiceGrpcTransport, transports.CatalogServiceGrpcAsyncIOTransport, transports.CatalogServiceRestTransport])
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
        while True:
            i = 10
    transport = CatalogServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CatalogServiceGrpcTransport)

def test_catalog_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CatalogServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_catalog_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.retail_v2.services.catalog_service.transports.CatalogServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CatalogServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_catalogs', 'update_catalog', 'set_default_branch', 'get_default_branch', 'get_completion_config', 'update_completion_config', 'get_attributes_config', 'update_attributes_config', 'add_catalog_attribute', 'remove_catalog_attribute', 'replace_catalog_attribute', 'get_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_catalog_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.retail_v2.services.catalog_service.transports.CatalogServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CatalogServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_catalog_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.retail_v2.services.catalog_service.transports.CatalogServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CatalogServiceTransport()
        adc.assert_called_once()

def test_catalog_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CatalogServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CatalogServiceGrpcTransport, transports.CatalogServiceGrpcAsyncIOTransport])
def test_catalog_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CatalogServiceGrpcTransport, transports.CatalogServiceGrpcAsyncIOTransport, transports.CatalogServiceRestTransport])
def test_catalog_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CatalogServiceGrpcTransport, grpc_helpers), (transports.CatalogServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_catalog_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CatalogServiceGrpcTransport, transports.CatalogServiceGrpcAsyncIOTransport])
def test_catalog_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

def test_catalog_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CatalogServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_catalog_service_host_no_port(transport_name):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_catalog_service_host_with_port(transport_name):
    if False:
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_catalog_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CatalogServiceClient(credentials=creds1, transport=transport_name)
    client2 = CatalogServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_catalogs._session
    session2 = client2.transport.list_catalogs._session
    assert session1 != session2
    session1 = client1.transport.update_catalog._session
    session2 = client2.transport.update_catalog._session
    assert session1 != session2
    session1 = client1.transport.set_default_branch._session
    session2 = client2.transport.set_default_branch._session
    assert session1 != session2
    session1 = client1.transport.get_default_branch._session
    session2 = client2.transport.get_default_branch._session
    assert session1 != session2
    session1 = client1.transport.get_completion_config._session
    session2 = client2.transport.get_completion_config._session
    assert session1 != session2
    session1 = client1.transport.update_completion_config._session
    session2 = client2.transport.update_completion_config._session
    assert session1 != session2
    session1 = client1.transport.get_attributes_config._session
    session2 = client2.transport.get_attributes_config._session
    assert session1 != session2
    session1 = client1.transport.update_attributes_config._session
    session2 = client2.transport.update_attributes_config._session
    assert session1 != session2
    session1 = client1.transport.add_catalog_attribute._session
    session2 = client2.transport.add_catalog_attribute._session
    assert session1 != session2
    session1 = client1.transport.remove_catalog_attribute._session
    session2 = client2.transport.remove_catalog_attribute._session
    assert session1 != session2
    session1 = client1.transport.replace_catalog_attribute._session
    session2 = client2.transport.replace_catalog_attribute._session
    assert session1 != session2

def test_catalog_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CatalogServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_catalog_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CatalogServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CatalogServiceGrpcTransport, transports.CatalogServiceGrpcAsyncIOTransport])
def test_catalog_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CatalogServiceGrpcTransport, transports.CatalogServiceGrpcAsyncIOTransport])
def test_catalog_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        print('Hello World!')
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

def test_attributes_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    catalog = 'whelk'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/attributesConfig'.format(project=project, location=location, catalog=catalog)
    actual = CatalogServiceClient.attributes_config_path(project, location, catalog)
    assert expected == actual

def test_parse_attributes_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'catalog': 'nudibranch'}
    path = CatalogServiceClient.attributes_config_path(**expected)
    actual = CatalogServiceClient.parse_attributes_config_path(path)
    assert expected == actual

def test_branch_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    catalog = 'winkle'
    branch = 'nautilus'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/branches/{branch}'.format(project=project, location=location, catalog=catalog, branch=branch)
    actual = CatalogServiceClient.branch_path(project, location, catalog, branch)
    assert expected == actual

def test_parse_branch_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone', 'catalog': 'squid', 'branch': 'clam'}
    path = CatalogServiceClient.branch_path(**expected)
    actual = CatalogServiceClient.parse_branch_path(path)
    assert expected == actual

def test_catalog_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    catalog = 'oyster'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}'.format(project=project, location=location, catalog=catalog)
    actual = CatalogServiceClient.catalog_path(project, location, catalog)
    assert expected == actual

def test_parse_catalog_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'catalog': 'mussel'}
    path = CatalogServiceClient.catalog_path(**expected)
    actual = CatalogServiceClient.parse_catalog_path(path)
    assert expected == actual

def test_completion_config_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    catalog = 'scallop'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/completionConfig'.format(project=project, location=location, catalog=catalog)
    actual = CatalogServiceClient.completion_config_path(project, location, catalog)
    assert expected == actual

def test_parse_completion_config_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone', 'location': 'squid', 'catalog': 'clam'}
    path = CatalogServiceClient.completion_config_path(**expected)
    actual = CatalogServiceClient.parse_completion_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CatalogServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'octopus'}
    path = CatalogServiceClient.common_billing_account_path(**expected)
    actual = CatalogServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CatalogServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nudibranch'}
    path = CatalogServiceClient.common_folder_path(**expected)
    actual = CatalogServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CatalogServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'mussel'}
    path = CatalogServiceClient.common_organization_path(**expected)
    actual = CatalogServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = CatalogServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus'}
    path = CatalogServiceClient.common_project_path(**expected)
    actual = CatalogServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CatalogServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'squid', 'location': 'clam'}
    path = CatalogServiceClient.common_location_path(**expected)
    actual = CatalogServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CatalogServiceTransport, '_prep_wrapped_messages') as prep:
        client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CatalogServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CatalogServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
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

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CatalogServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CatalogServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CatalogServiceClient, transports.CatalogServiceGrpcTransport), (CatalogServiceAsyncClient, transports.CatalogServiceGrpcAsyncIOTransport)])
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