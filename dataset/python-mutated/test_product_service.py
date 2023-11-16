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
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.type import date_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.retail_v2.services.product_service import ProductServiceAsyncClient, ProductServiceClient, pagers, transports
from google.cloud.retail_v2.types import common, import_config
from google.cloud.retail_v2.types import product
from google.cloud.retail_v2.types import product as gcr_product
from google.cloud.retail_v2.types import product_service, promotion

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert ProductServiceClient._get_default_mtls_endpoint(None) is None
    assert ProductServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ProductServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ProductServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ProductServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ProductServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ProductServiceClient, 'grpc'), (ProductServiceAsyncClient, 'grpc_asyncio'), (ProductServiceClient, 'rest')])
def test_product_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ProductServiceGrpcTransport, 'grpc'), (transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ProductServiceRestTransport, 'rest')])
def test_product_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ProductServiceClient, 'grpc'), (ProductServiceAsyncClient, 'grpc_asyncio'), (ProductServiceClient, 'rest')])
def test_product_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

def test_product_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ProductServiceClient.get_transport_class()
    available_transports = [transports.ProductServiceGrpcTransport, transports.ProductServiceRestTransport]
    assert transport in available_transports
    transport = ProductServiceClient.get_transport_class('grpc')
    assert transport == transports.ProductServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ProductServiceClient, transports.ProductServiceGrpcTransport, 'grpc'), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ProductServiceClient, transports.ProductServiceRestTransport, 'rest')])
@mock.patch.object(ProductServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ProductServiceClient))
@mock.patch.object(ProductServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ProductServiceAsyncClient))
def test_product_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(ProductServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ProductServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ProductServiceClient, transports.ProductServiceGrpcTransport, 'grpc', 'true'), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ProductServiceClient, transports.ProductServiceGrpcTransport, 'grpc', 'false'), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ProductServiceClient, transports.ProductServiceRestTransport, 'rest', 'true'), (ProductServiceClient, transports.ProductServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ProductServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ProductServiceClient))
@mock.patch.object(ProductServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ProductServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_product_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ProductServiceClient, ProductServiceAsyncClient])
@mock.patch.object(ProductServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ProductServiceClient))
@mock.patch.object(ProductServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ProductServiceAsyncClient))
def test_product_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ProductServiceClient, transports.ProductServiceGrpcTransport, 'grpc'), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ProductServiceClient, transports.ProductServiceRestTransport, 'rest')])
def test_product_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ProductServiceClient, transports.ProductServiceGrpcTransport, 'grpc', grpc_helpers), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ProductServiceClient, transports.ProductServiceRestTransport, 'rest', None)])
def test_product_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_product_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.retail_v2.services.product_service.transports.ProductServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ProductServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ProductServiceClient, transports.ProductServiceGrpcTransport, 'grpc', grpc_helpers), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_product_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [product_service.CreateProductRequest, dict])
def test_create_product(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        call.return_value = gcr_product.Product(name='name_value', id='id_value', type_=gcr_product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=gcr_product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value'])
        response = client.create_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.CreateProductRequest()
    assert isinstance(response, gcr_product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == gcr_product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == gcr_product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

def test_create_product_empty_call():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        client.create_product()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.CreateProductRequest()

@pytest.mark.asyncio
async def test_create_product_async(transport: str='grpc_asyncio', request_type=product_service.CreateProductRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_product.Product(name='name_value', id='id_value', type_=gcr_product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=gcr_product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value']))
        response = await client.create_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.CreateProductRequest()
    assert isinstance(response, gcr_product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == gcr_product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == gcr_product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

@pytest.mark.asyncio
async def test_create_product_async_from_dict():
    await test_create_product_async(request_type=dict)

def test_create_product_field_headers():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.CreateProductRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        call.return_value = gcr_product.Product()
        client.create_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_product_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.CreateProductRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_product.Product())
        await client.create_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_product_flattened():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        call.return_value = gcr_product.Product()
        client.create_product(parent='parent_value', product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), product_id='product_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].product
        mock_val = gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].product_id
        mock_val = 'product_id_value'
        assert arg == mock_val

def test_create_product_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_product(product_service.CreateProductRequest(), parent='parent_value', product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), product_id='product_id_value')

@pytest.mark.asyncio
async def test_create_product_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_product), '__call__') as call:
        call.return_value = gcr_product.Product()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_product.Product())
        response = await client.create_product(parent='parent_value', product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), product_id='product_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].product
        mock_val = gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].product_id
        mock_val = 'product_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_product_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_product(product_service.CreateProductRequest(), parent='parent_value', product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), product_id='product_id_value')

@pytest.mark.parametrize('request_type', [product_service.GetProductRequest, dict])
def test_get_product(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        call.return_value = product.Product(name='name_value', id='id_value', type_=product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value'])
        response = client.get_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.GetProductRequest()
    assert isinstance(response, product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

def test_get_product_empty_call():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        client.get_product()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.GetProductRequest()

@pytest.mark.asyncio
async def test_get_product_async(transport: str='grpc_asyncio', request_type=product_service.GetProductRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(product.Product(name='name_value', id='id_value', type_=product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value']))
        response = await client.get_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.GetProductRequest()
    assert isinstance(response, product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

@pytest.mark.asyncio
async def test_get_product_async_from_dict():
    await test_get_product_async(request_type=dict)

def test_get_product_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.GetProductRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        call.return_value = product.Product()
        client.get_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_product_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.GetProductRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(product.Product())
        await client.get_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_product_flattened():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        call.return_value = product.Product()
        client.get_product(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_product_flattened_error():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_product(product_service.GetProductRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_product_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_product), '__call__') as call:
        call.return_value = product.Product()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(product.Product())
        response = await client.get_product(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_product_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_product(product_service.GetProductRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [product_service.ListProductsRequest, dict])
def test_list_products(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = product_service.ListProductsResponse(next_page_token='next_page_token_value')
        response = client.list_products(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.ListProductsRequest()
    assert isinstance(response, pagers.ListProductsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_products_empty_call():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        client.list_products()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.ListProductsRequest()

@pytest.mark.asyncio
async def test_list_products_async(transport: str='grpc_asyncio', request_type=product_service.ListProductsRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(product_service.ListProductsResponse(next_page_token='next_page_token_value'))
        response = await client.list_products(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.ListProductsRequest()
    assert isinstance(response, pagers.ListProductsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_products_async_from_dict():
    await test_list_products_async(request_type=dict)

def test_list_products_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.ListProductsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = product_service.ListProductsResponse()
        client.list_products(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_products_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.ListProductsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(product_service.ListProductsResponse())
        await client.list_products(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_products_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = product_service.ListProductsResponse()
        client.list_products(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_products_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_products(product_service.ListProductsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_products_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = product_service.ListProductsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(product_service.ListProductsResponse())
        response = await client.list_products(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_products_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_products(product_service.ListProductsRequest(), parent='parent_value')

def test_list_products_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.side_effect = (product_service.ListProductsResponse(products=[product.Product(), product.Product(), product.Product()], next_page_token='abc'), product_service.ListProductsResponse(products=[], next_page_token='def'), product_service.ListProductsResponse(products=[product.Product()], next_page_token='ghi'), product_service.ListProductsResponse(products=[product.Product(), product.Product()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_products(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, product.Product) for i in results))

def test_list_products_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.side_effect = (product_service.ListProductsResponse(products=[product.Product(), product.Product(), product.Product()], next_page_token='abc'), product_service.ListProductsResponse(products=[], next_page_token='def'), product_service.ListProductsResponse(products=[product.Product()], next_page_token='ghi'), product_service.ListProductsResponse(products=[product.Product(), product.Product()]), RuntimeError)
        pages = list(client.list_products(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_products_async_pager():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_products), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (product_service.ListProductsResponse(products=[product.Product(), product.Product(), product.Product()], next_page_token='abc'), product_service.ListProductsResponse(products=[], next_page_token='def'), product_service.ListProductsResponse(products=[product.Product()], next_page_token='ghi'), product_service.ListProductsResponse(products=[product.Product(), product.Product()]), RuntimeError)
        async_pager = await client.list_products(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, product.Product) for i in responses))

@pytest.mark.asyncio
async def test_list_products_async_pages():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_products), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (product_service.ListProductsResponse(products=[product.Product(), product.Product(), product.Product()], next_page_token='abc'), product_service.ListProductsResponse(products=[], next_page_token='def'), product_service.ListProductsResponse(products=[product.Product()], next_page_token='ghi'), product_service.ListProductsResponse(products=[product.Product(), product.Product()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_products(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [product_service.UpdateProductRequest, dict])
def test_update_product(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        call.return_value = gcr_product.Product(name='name_value', id='id_value', type_=gcr_product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=gcr_product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value'])
        response = client.update_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.UpdateProductRequest()
    assert isinstance(response, gcr_product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == gcr_product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == gcr_product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

def test_update_product_empty_call():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        client.update_product()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.UpdateProductRequest()

@pytest.mark.asyncio
async def test_update_product_async(transport: str='grpc_asyncio', request_type=product_service.UpdateProductRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_product.Product(name='name_value', id='id_value', type_=gcr_product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=gcr_product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value']))
        response = await client.update_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.UpdateProductRequest()
    assert isinstance(response, gcr_product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == gcr_product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == gcr_product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

@pytest.mark.asyncio
async def test_update_product_async_from_dict():
    await test_update_product_async(request_type=dict)

def test_update_product_field_headers():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.UpdateProductRequest()
    request.product.name = 'name_value'
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        call.return_value = gcr_product.Product()
        client.update_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_product_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.UpdateProductRequest()
    request.product.name = 'name_value'
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_product.Product())
        await client.update_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product.name=name_value') in kw['metadata']

def test_update_product_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        call.return_value = gcr_product.Product()
        client.update_product(product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_product_flattened_error():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_product(product_service.UpdateProductRequest(), product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_product_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_product), '__call__') as call:
        call.return_value = gcr_product.Product()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_product.Product())
        response = await client.update_product(product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_product_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_product(product_service.UpdateProductRequest(), product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [product_service.DeleteProductRequest, dict])
def test_delete_product(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        call.return_value = None
        response = client.delete_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.DeleteProductRequest()
    assert response is None

def test_delete_product_empty_call():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        client.delete_product()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.DeleteProductRequest()

@pytest.mark.asyncio
async def test_delete_product_async(transport: str='grpc_asyncio', request_type=product_service.DeleteProductRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.DeleteProductRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_product_async_from_dict():
    await test_delete_product_async(request_type=dict)

def test_delete_product_field_headers():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.DeleteProductRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        call.return_value = None
        client.delete_product(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_product_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.DeleteProductRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_product(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_product_flattened():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        call.return_value = None
        client.delete_product(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_product_flattened_error():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_product(product_service.DeleteProductRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_product_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_product), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_product(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_product_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_product(product_service.DeleteProductRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [import_config.ImportProductsRequest, dict])
def test_import_products(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_products), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.import_products(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == import_config.ImportProductsRequest()
    assert isinstance(response, future.Future)

def test_import_products_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_products), '__call__') as call:
        client.import_products()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == import_config.ImportProductsRequest()

@pytest.mark.asyncio
async def test_import_products_async(transport: str='grpc_asyncio', request_type=import_config.ImportProductsRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_products), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_products(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == import_config.ImportProductsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_import_products_async_from_dict():
    await test_import_products_async(request_type=dict)

def test_import_products_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = import_config.ImportProductsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_products), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_products(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_products_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = import_config.ImportProductsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_products), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.import_products(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [product_service.SetInventoryRequest, dict])
def test_set_inventory(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.set_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.SetInventoryRequest()
    assert isinstance(response, future.Future)

def test_set_inventory_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        client.set_inventory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.SetInventoryRequest()

@pytest.mark.asyncio
async def test_set_inventory_async(transport: str='grpc_asyncio', request_type=product_service.SetInventoryRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.set_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.SetInventoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_set_inventory_async_from_dict():
    await test_set_inventory_async(request_type=dict)

def test_set_inventory_field_headers():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.SetInventoryRequest()
    request.inventory.name = 'name_value'
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.set_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'inventory.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_inventory_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.SetInventoryRequest()
    request.inventory.name = 'name_value'
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.set_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'inventory.name=name_value') in kw['metadata']

def test_set_inventory_flattened():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.set_inventory(inventory=product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), set_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].inventory
        mock_val = product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].set_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_set_inventory_flattened_error():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_inventory(product_service.SetInventoryRequest(), inventory=product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), set_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_set_inventory_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_inventory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.set_inventory(inventory=product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), set_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].inventory
        mock_val = product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751))
        assert arg == mock_val
        arg = args[0].set_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_inventory_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_inventory(product_service.SetInventoryRequest(), inventory=product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), set_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [product_service.AddFulfillmentPlacesRequest, dict])
def test_add_fulfillment_places(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.add_fulfillment_places(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.AddFulfillmentPlacesRequest()
    assert isinstance(response, future.Future)

def test_add_fulfillment_places_empty_call():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        client.add_fulfillment_places()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.AddFulfillmentPlacesRequest()

@pytest.mark.asyncio
async def test_add_fulfillment_places_async(transport: str='grpc_asyncio', request_type=product_service.AddFulfillmentPlacesRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_fulfillment_places(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.AddFulfillmentPlacesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_add_fulfillment_places_async_from_dict():
    await test_add_fulfillment_places_async(request_type=dict)

def test_add_fulfillment_places_field_headers():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.AddFulfillmentPlacesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_fulfillment_places(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

@pytest.mark.asyncio
async def test_add_fulfillment_places_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.AddFulfillmentPlacesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.add_fulfillment_places(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

def test_add_fulfillment_places_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_fulfillment_places(product='product_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

def test_add_fulfillment_places_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.add_fulfillment_places(product_service.AddFulfillmentPlacesRequest(), product='product_value')

@pytest.mark.asyncio
async def test_add_fulfillment_places_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_fulfillment_places(product='product_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_add_fulfillment_places_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.add_fulfillment_places(product_service.AddFulfillmentPlacesRequest(), product='product_value')

@pytest.mark.parametrize('request_type', [product_service.RemoveFulfillmentPlacesRequest, dict])
def test_remove_fulfillment_places(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.remove_fulfillment_places(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.RemoveFulfillmentPlacesRequest()
    assert isinstance(response, future.Future)

def test_remove_fulfillment_places_empty_call():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        client.remove_fulfillment_places()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.RemoveFulfillmentPlacesRequest()

@pytest.mark.asyncio
async def test_remove_fulfillment_places_async(transport: str='grpc_asyncio', request_type=product_service.RemoveFulfillmentPlacesRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_fulfillment_places(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.RemoveFulfillmentPlacesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_remove_fulfillment_places_async_from_dict():
    await test_remove_fulfillment_places_async(request_type=dict)

def test_remove_fulfillment_places_field_headers():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.RemoveFulfillmentPlacesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_fulfillment_places(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

@pytest.mark.asyncio
async def test_remove_fulfillment_places_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.RemoveFulfillmentPlacesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.remove_fulfillment_places(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

def test_remove_fulfillment_places_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_fulfillment_places(product='product_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

def test_remove_fulfillment_places_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.remove_fulfillment_places(product_service.RemoveFulfillmentPlacesRequest(), product='product_value')

@pytest.mark.asyncio
async def test_remove_fulfillment_places_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.remove_fulfillment_places), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_fulfillment_places(product='product_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_remove_fulfillment_places_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.remove_fulfillment_places(product_service.RemoveFulfillmentPlacesRequest(), product='product_value')

@pytest.mark.parametrize('request_type', [product_service.AddLocalInventoriesRequest, dict])
def test_add_local_inventories(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.add_local_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.AddLocalInventoriesRequest()
    assert isinstance(response, future.Future)

def test_add_local_inventories_empty_call():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        client.add_local_inventories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.AddLocalInventoriesRequest()

@pytest.mark.asyncio
async def test_add_local_inventories_async(transport: str='grpc_asyncio', request_type=product_service.AddLocalInventoriesRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_local_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.AddLocalInventoriesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_add_local_inventories_async_from_dict():
    await test_add_local_inventories_async(request_type=dict)

def test_add_local_inventories_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.AddLocalInventoriesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_local_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

@pytest.mark.asyncio
async def test_add_local_inventories_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.AddLocalInventoriesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.add_local_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

def test_add_local_inventories_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_local_inventories(product='product_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

def test_add_local_inventories_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.add_local_inventories(product_service.AddLocalInventoriesRequest(), product='product_value')

@pytest.mark.asyncio
async def test_add_local_inventories_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_local_inventories(product='product_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_add_local_inventories_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.add_local_inventories(product_service.AddLocalInventoriesRequest(), product='product_value')

@pytest.mark.parametrize('request_type', [product_service.RemoveLocalInventoriesRequest, dict])
def test_remove_local_inventories(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.remove_local_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.RemoveLocalInventoriesRequest()
    assert isinstance(response, future.Future)

def test_remove_local_inventories_empty_call():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        client.remove_local_inventories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.RemoveLocalInventoriesRequest()

@pytest.mark.asyncio
async def test_remove_local_inventories_async(transport: str='grpc_asyncio', request_type=product_service.RemoveLocalInventoriesRequest):
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_local_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == product_service.RemoveLocalInventoriesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_remove_local_inventories_async_from_dict():
    await test_remove_local_inventories_async(request_type=dict)

def test_remove_local_inventories_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.RemoveLocalInventoriesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_local_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

@pytest.mark.asyncio
async def test_remove_local_inventories_field_headers_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = product_service.RemoveLocalInventoriesRequest()
    request.product = 'product_value'
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.remove_local_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'product=product_value') in kw['metadata']

def test_remove_local_inventories_flattened():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_local_inventories(product='product_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

def test_remove_local_inventories_flattened_error():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.remove_local_inventories(product_service.RemoveLocalInventoriesRequest(), product='product_value')

@pytest.mark.asyncio
async def test_remove_local_inventories_flattened_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.remove_local_inventories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_local_inventories(product='product_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].product
        mock_val = 'product_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_remove_local_inventories_flattened_error_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.remove_local_inventories(product_service.RemoveLocalInventoriesRequest(), product='product_value')

@pytest.mark.parametrize('request_type', [product_service.CreateProductRequest, dict])
def test_create_product_rest(request_type):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
    request_init['product'] = {'expire_time': {'seconds': 751, 'nanos': 543}, 'ttl': {'seconds': 751, 'nanos': 543}, 'name': 'name_value', 'id': 'id_value', 'type_': 1, 'primary_product_id': 'primary_product_id_value', 'collection_member_ids': ['collection_member_ids_value1', 'collection_member_ids_value2'], 'gtin': 'gtin_value', 'categories': ['categories_value1', 'categories_value2'], 'title': 'title_value', 'brands': ['brands_value1', 'brands_value2'], 'description': 'description_value', 'language_code': 'language_code_value', 'attributes': {}, 'tags': ['tags_value1', 'tags_value2'], 'price_info': {'currency_code': 'currency_code_value', 'price': 0.531, 'original_price': 0.1479, 'cost': 0.441, 'price_effective_time': {}, 'price_expire_time': {}, 'price_range': {'price': {'minimum': 0.764, 'exclusive_minimum': 0.18430000000000002, 'maximum': 0.766, 'exclusive_maximum': 0.1845}, 'original_price': {}}}, 'rating': {'rating_count': 1293, 'average_rating': 0.1471, 'rating_histogram': [1715, 1716]}, 'available_time': {}, 'availability': 1, 'available_quantity': {'value': 541}, 'fulfillment_info': [{'type_': 'type__value', 'place_ids': ['place_ids_value1', 'place_ids_value2']}], 'uri': 'uri_value', 'images': [{'uri': 'uri_value', 'height': 633, 'width': 544}], 'audience': {'genders': ['genders_value1', 'genders_value2'], 'age_groups': ['age_groups_value1', 'age_groups_value2']}, 'color_info': {'color_families': ['color_families_value1', 'color_families_value2'], 'colors': ['colors_value1', 'colors_value2']}, 'sizes': ['sizes_value1', 'sizes_value2'], 'materials': ['materials_value1', 'materials_value2'], 'patterns': ['patterns_value1', 'patterns_value2'], 'conditions': ['conditions_value1', 'conditions_value2'], 'promotions': [{'promotion_id': 'promotion_id_value'}], 'publish_time': {}, 'retrievable_fields': {'paths': ['paths_value1', 'paths_value2']}, 'variants': {}, 'local_inventories': [{'place_id': 'place_id_value', 'price_info': {}, 'attributes': {}, 'fulfillment_types': ['fulfillment_types_value1', 'fulfillment_types_value2']}]}
    test_field = product_service.CreateProductRequest.meta.fields['product']

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
    for (field, value) in request_init['product'].items():
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
                for i in range(0, len(request_init['product'][field])):
                    del request_init['product'][field][i][subfield]
            else:
                del request_init['product'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_product.Product(name='name_value', id='id_value', type_=gcr_product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=gcr_product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_product.Product.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_product(request)
    assert isinstance(response, gcr_product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == gcr_product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == gcr_product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

def test_create_product_rest_required_fields(request_type=product_service.CreateProductRequest):
    if False:
        return 10
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['product_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'productId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_product._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'productId' in jsonified_request
    assert jsonified_request['productId'] == request_init['product_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['productId'] = 'product_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_product._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('product_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'productId' in jsonified_request
    assert jsonified_request['productId'] == 'product_id_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcr_product.Product()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcr_product.Product.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_product(request)
            expected_params = [('productId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_product_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_product._get_unset_required_fields({})
    assert set(unset_fields) == set(('productId',)) & set(('parent', 'product', 'productId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_product_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ProductServiceRestInterceptor, 'post_create_product') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_create_product') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.CreateProductRequest.pb(product_service.CreateProductRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcr_product.Product.to_json(gcr_product.Product())
        request = product_service.CreateProductRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcr_product.Product()
        client.create_product(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_product_rest_bad_request(transport: str='rest', request_type=product_service.CreateProductRequest):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_product(request)

def test_create_product_rest_flattened():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_product.Product()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
        mock_args = dict(parent='parent_value', product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), product_id='product_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_product.Product.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_product(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/catalogs/*/branches/*}/products' % client.transport._host, args[1])

def test_create_product_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_product(product_service.CreateProductRequest(), parent='parent_value', product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), product_id='product_id_value')

def test_create_product_rest_error():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.GetProductRequest, dict])
def test_get_product_rest(request_type):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = product.Product(name='name_value', id='id_value', type_=product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = product.Product.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_product(request)
    assert isinstance(response, product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

def test_get_product_rest_required_fields(request_type=product_service.GetProductRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_product._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_product._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = product.Product()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = product.Product.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_product(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_product_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_product._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_product_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ProductServiceRestInterceptor, 'post_get_product') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_get_product') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.GetProductRequest.pb(product_service.GetProductRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = product.Product.to_json(product.Product())
        request = product_service.GetProductRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = product.Product()
        client.get_product(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_product_rest_bad_request(transport: str='rest', request_type=product_service.GetProductRequest):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_product(request)

def test_get_product_rest_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = product.Product()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = product.Product.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_product(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/catalogs/*/branches/*/products/**}' % client.transport._host, args[1])

def test_get_product_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_product(product_service.GetProductRequest(), name='name_value')

def test_get_product_rest_error():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.ListProductsRequest, dict])
def test_list_products_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = product_service.ListProductsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = product_service.ListProductsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_products(request)
    assert isinstance(response, pagers.ListProductsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_products_rest_required_fields(request_type=product_service.ListProductsRequest):
    if False:
        return 10
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_products._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_products._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token', 'read_mask'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = product_service.ListProductsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = product_service.ListProductsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_products(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_products_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_products._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken', 'readMask')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_products_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ProductServiceRestInterceptor, 'post_list_products') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_list_products') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.ListProductsRequest.pb(product_service.ListProductsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = product_service.ListProductsResponse.to_json(product_service.ListProductsResponse())
        request = product_service.ListProductsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = product_service.ListProductsResponse()
        client.list_products(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_products_rest_bad_request(transport: str='rest', request_type=product_service.ListProductsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_products(request)

def test_list_products_rest_flattened():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = product_service.ListProductsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = product_service.ListProductsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_products(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/catalogs/*/branches/*}/products' % client.transport._host, args[1])

def test_list_products_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_products(product_service.ListProductsRequest(), parent='parent_value')

def test_list_products_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (product_service.ListProductsResponse(products=[product.Product(), product.Product(), product.Product()], next_page_token='abc'), product_service.ListProductsResponse(products=[], next_page_token='def'), product_service.ListProductsResponse(products=[product.Product()], next_page_token='ghi'), product_service.ListProductsResponse(products=[product.Product(), product.Product()]))
        response = response + response
        response = tuple((product_service.ListProductsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
        pager = client.list_products(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, product.Product) for i in results))
        pages = list(client.list_products(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [product_service.UpdateProductRequest, dict])
def test_update_product_rest(request_type):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'product': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}}
    request_init['product'] = {'expire_time': {'seconds': 751, 'nanos': 543}, 'ttl': {'seconds': 751, 'nanos': 543}, 'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5', 'id': 'id_value', 'type_': 1, 'primary_product_id': 'primary_product_id_value', 'collection_member_ids': ['collection_member_ids_value1', 'collection_member_ids_value2'], 'gtin': 'gtin_value', 'categories': ['categories_value1', 'categories_value2'], 'title': 'title_value', 'brands': ['brands_value1', 'brands_value2'], 'description': 'description_value', 'language_code': 'language_code_value', 'attributes': {}, 'tags': ['tags_value1', 'tags_value2'], 'price_info': {'currency_code': 'currency_code_value', 'price': 0.531, 'original_price': 0.1479, 'cost': 0.441, 'price_effective_time': {}, 'price_expire_time': {}, 'price_range': {'price': {'minimum': 0.764, 'exclusive_minimum': 0.18430000000000002, 'maximum': 0.766, 'exclusive_maximum': 0.1845}, 'original_price': {}}}, 'rating': {'rating_count': 1293, 'average_rating': 0.1471, 'rating_histogram': [1715, 1716]}, 'available_time': {}, 'availability': 1, 'available_quantity': {'value': 541}, 'fulfillment_info': [{'type_': 'type__value', 'place_ids': ['place_ids_value1', 'place_ids_value2']}], 'uri': 'uri_value', 'images': [{'uri': 'uri_value', 'height': 633, 'width': 544}], 'audience': {'genders': ['genders_value1', 'genders_value2'], 'age_groups': ['age_groups_value1', 'age_groups_value2']}, 'color_info': {'color_families': ['color_families_value1', 'color_families_value2'], 'colors': ['colors_value1', 'colors_value2']}, 'sizes': ['sizes_value1', 'sizes_value2'], 'materials': ['materials_value1', 'materials_value2'], 'patterns': ['patterns_value1', 'patterns_value2'], 'conditions': ['conditions_value1', 'conditions_value2'], 'promotions': [{'promotion_id': 'promotion_id_value'}], 'publish_time': {}, 'retrievable_fields': {'paths': ['paths_value1', 'paths_value2']}, 'variants': {}, 'local_inventories': [{'place_id': 'place_id_value', 'price_info': {}, 'attributes': {}, 'fulfillment_types': ['fulfillment_types_value1', 'fulfillment_types_value2']}]}
    test_field = product_service.UpdateProductRequest.meta.fields['product']

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
    for (field, value) in request_init['product'].items():
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
                for i in range(0, len(request_init['product'][field])):
                    del request_init['product'][field][i][subfield]
            else:
                del request_init['product'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_product.Product(name='name_value', id='id_value', type_=gcr_product.Product.Type.PRIMARY, primary_product_id='primary_product_id_value', collection_member_ids=['collection_member_ids_value'], gtin='gtin_value', categories=['categories_value'], title='title_value', brands=['brands_value'], description='description_value', language_code='language_code_value', tags=['tags_value'], availability=gcr_product.Product.Availability.IN_STOCK, uri='uri_value', sizes=['sizes_value'], materials=['materials_value'], patterns=['patterns_value'], conditions=['conditions_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_product.Product.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_product(request)
    assert isinstance(response, gcr_product.Product)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == gcr_product.Product.Type.PRIMARY
    assert response.primary_product_id == 'primary_product_id_value'
    assert response.collection_member_ids == ['collection_member_ids_value']
    assert response.gtin == 'gtin_value'
    assert response.categories == ['categories_value']
    assert response.title == 'title_value'
    assert response.brands == ['brands_value']
    assert response.description == 'description_value'
    assert response.language_code == 'language_code_value'
    assert response.tags == ['tags_value']
    assert response.availability == gcr_product.Product.Availability.IN_STOCK
    assert response.uri == 'uri_value'
    assert response.sizes == ['sizes_value']
    assert response.materials == ['materials_value']
    assert response.patterns == ['patterns_value']
    assert response.conditions == ['conditions_value']

def test_update_product_rest_required_fields(request_type=product_service.UpdateProductRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_product._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_product._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcr_product.Product()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcr_product.Product.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_product(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_product_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_product._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'updateMask')) & set(('product',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_product_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ProductServiceRestInterceptor, 'post_update_product') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_update_product') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.UpdateProductRequest.pb(product_service.UpdateProductRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcr_product.Product.to_json(gcr_product.Product())
        request = product_service.UpdateProductRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcr_product.Product()
        client.update_product(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_product_rest_bad_request(transport: str='rest', request_type=product_service.UpdateProductRequest):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'product': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_product(request)

def test_update_product_rest_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_product.Product()
        sample_request = {'product': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}}
        mock_args = dict(product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_product.Product.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_product(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{product.name=projects/*/locations/*/catalogs/*/branches/*/products/**}' % client.transport._host, args[1])

def test_update_product_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_product(product_service.UpdateProductRequest(), product=gcr_product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_product_rest_error():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.DeleteProductRequest, dict])
def test_delete_product_rest(request_type):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_product(request)
    assert response is None

def test_delete_product_rest_required_fields(request_type=product_service.DeleteProductRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_product._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_product._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_product(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_product_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_product._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_product_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_delete_product') as pre:
        pre.assert_not_called()
        pb_message = product_service.DeleteProductRequest.pb(product_service.DeleteProductRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = product_service.DeleteProductRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_product(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_product_rest_bad_request(transport: str='rest', request_type=product_service.DeleteProductRequest):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_product(request)

def test_delete_product_rest_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_product(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/catalogs/*/branches/*/products/**}' % client.transport._host, args[1])

def test_delete_product_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_product(product_service.DeleteProductRequest(), name='name_value')

def test_delete_product_rest_error():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [import_config.ImportProductsRequest, dict])
def test_import_products_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_products(request)
    assert response.operation.name == 'operations/spam'

def test_import_products_rest_required_fields(request_type=import_config.ImportProductsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_products._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_products._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.import_products(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_products_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_products._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'inputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_products_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ProductServiceRestInterceptor, 'post_import_products') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_import_products') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = import_config.ImportProductsRequest.pb(import_config.ImportProductsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = import_config.ImportProductsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.import_products(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_products_rest_bad_request(transport: str='rest', request_type=import_config.ImportProductsRequest):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_products(request)

def test_import_products_rest_error():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.SetInventoryRequest, dict])
def test_set_inventory_rest(request_type):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'inventory': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_inventory(request)
    assert response.operation.name == 'operations/spam'

def test_set_inventory_rest_required_fields(request_type=product_service.SetInventoryRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_inventory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_inventory_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_inventory._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('inventory',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_inventory_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ProductServiceRestInterceptor, 'post_set_inventory') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_set_inventory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.SetInventoryRequest.pb(product_service.SetInventoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = product_service.SetInventoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.set_inventory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_inventory_rest_bad_request(transport: str='rest', request_type=product_service.SetInventoryRequest):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'inventory': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_inventory(request)

def test_set_inventory_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'inventory': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}}
        mock_args = dict(inventory=product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), set_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_inventory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{inventory.name=projects/*/locations/*/catalogs/*/branches/*/products/**}:setInventory' % client.transport._host, args[1])

def test_set_inventory_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_inventory(product_service.SetInventoryRequest(), inventory=product.Product(expire_time=timestamp_pb2.Timestamp(seconds=751)), set_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_set_inventory_rest_error():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.AddFulfillmentPlacesRequest, dict])
def test_add_fulfillment_places_rest(request_type):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_fulfillment_places(request)
    assert response.operation.name == 'operations/spam'

def test_add_fulfillment_places_rest_required_fields(request_type=product_service.AddFulfillmentPlacesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['product'] = ''
    request_init['type_'] = ''
    request_init['place_ids'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_fulfillment_places._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['product'] = 'product_value'
    jsonified_request['type'] = 'type__value'
    jsonified_request['placeIds'] = 'place_ids_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_fulfillment_places._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'product' in jsonified_request
    assert jsonified_request['product'] == 'product_value'
    assert 'type' in jsonified_request
    assert jsonified_request['type'] == 'type__value'
    assert 'placeIds' in jsonified_request
    assert jsonified_request['placeIds'] == 'place_ids_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_fulfillment_places(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_fulfillment_places_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_fulfillment_places._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('product', 'type', 'placeIds'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_fulfillment_places_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ProductServiceRestInterceptor, 'post_add_fulfillment_places') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_add_fulfillment_places') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.AddFulfillmentPlacesRequest.pb(product_service.AddFulfillmentPlacesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = product_service.AddFulfillmentPlacesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.add_fulfillment_places(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_fulfillment_places_rest_bad_request(transport: str='rest', request_type=product_service.AddFulfillmentPlacesRequest):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_fulfillment_places(request)

def test_add_fulfillment_places_rest_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
        mock_args = dict(product='product_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_fulfillment_places(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{product=projects/*/locations/*/catalogs/*/branches/*/products/**}:addFulfillmentPlaces' % client.transport._host, args[1])

def test_add_fulfillment_places_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_fulfillment_places(product_service.AddFulfillmentPlacesRequest(), product='product_value')

def test_add_fulfillment_places_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.RemoveFulfillmentPlacesRequest, dict])
def test_remove_fulfillment_places_rest(request_type):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_fulfillment_places(request)
    assert response.operation.name == 'operations/spam'

def test_remove_fulfillment_places_rest_required_fields(request_type=product_service.RemoveFulfillmentPlacesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['product'] = ''
    request_init['type_'] = ''
    request_init['place_ids'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_fulfillment_places._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['product'] = 'product_value'
    jsonified_request['type'] = 'type__value'
    jsonified_request['placeIds'] = 'place_ids_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_fulfillment_places._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'product' in jsonified_request
    assert jsonified_request['product'] == 'product_value'
    assert 'type' in jsonified_request
    assert jsonified_request['type'] == 'type__value'
    assert 'placeIds' in jsonified_request
    assert jsonified_request['placeIds'] == 'place_ids_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_fulfillment_places(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_fulfillment_places_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_fulfillment_places._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('product', 'type', 'placeIds'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_fulfillment_places_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ProductServiceRestInterceptor, 'post_remove_fulfillment_places') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_remove_fulfillment_places') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.RemoveFulfillmentPlacesRequest.pb(product_service.RemoveFulfillmentPlacesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = product_service.RemoveFulfillmentPlacesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.remove_fulfillment_places(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_fulfillment_places_rest_bad_request(transport: str='rest', request_type=product_service.RemoveFulfillmentPlacesRequest):
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_fulfillment_places(request)

def test_remove_fulfillment_places_rest_flattened():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
        mock_args = dict(product='product_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_fulfillment_places(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{product=projects/*/locations/*/catalogs/*/branches/*/products/**}:removeFulfillmentPlaces' % client.transport._host, args[1])

def test_remove_fulfillment_places_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_fulfillment_places(product_service.RemoveFulfillmentPlacesRequest(), product='product_value')

def test_remove_fulfillment_places_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.AddLocalInventoriesRequest, dict])
def test_add_local_inventories_rest(request_type):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_local_inventories(request)
    assert response.operation.name == 'operations/spam'

def test_add_local_inventories_rest_required_fields(request_type=product_service.AddLocalInventoriesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['product'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_local_inventories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['product'] = 'product_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_local_inventories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'product' in jsonified_request
    assert jsonified_request['product'] == 'product_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_local_inventories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_local_inventories_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_local_inventories._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('product', 'localInventories'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_local_inventories_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ProductServiceRestInterceptor, 'post_add_local_inventories') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_add_local_inventories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.AddLocalInventoriesRequest.pb(product_service.AddLocalInventoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = product_service.AddLocalInventoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.add_local_inventories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_local_inventories_rest_bad_request(transport: str='rest', request_type=product_service.AddLocalInventoriesRequest):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_local_inventories(request)

def test_add_local_inventories_rest_flattened():
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
        mock_args = dict(product='product_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_local_inventories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{product=projects/*/locations/*/catalogs/*/branches/*/products/**}:addLocalInventories' % client.transport._host, args[1])

def test_add_local_inventories_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_local_inventories(product_service.AddLocalInventoriesRequest(), product='product_value')

def test_add_local_inventories_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [product_service.RemoveLocalInventoriesRequest, dict])
def test_remove_local_inventories_rest(request_type):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_local_inventories(request)
    assert response.operation.name == 'operations/spam'

def test_remove_local_inventories_rest_required_fields(request_type=product_service.RemoveLocalInventoriesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ProductServiceRestTransport
    request_init = {}
    request_init['product'] = ''
    request_init['place_ids'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_local_inventories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['product'] = 'product_value'
    jsonified_request['placeIds'] = 'place_ids_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_local_inventories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'product' in jsonified_request
    assert jsonified_request['product'] == 'product_value'
    assert 'placeIds' in jsonified_request
    assert jsonified_request['placeIds'] == 'place_ids_value'
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_local_inventories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_local_inventories_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_local_inventories._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('product', 'placeIds'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_local_inventories_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ProductServiceRestInterceptor())
    client = ProductServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ProductServiceRestInterceptor, 'post_remove_local_inventories') as post, mock.patch.object(transports.ProductServiceRestInterceptor, 'pre_remove_local_inventories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = product_service.RemoveLocalInventoriesRequest.pb(product_service.RemoveLocalInventoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = product_service.RemoveLocalInventoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.remove_local_inventories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_local_inventories_rest_bad_request(transport: str='rest', request_type=product_service.RemoveLocalInventoriesRequest):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_local_inventories(request)

def test_remove_local_inventories_rest_flattened():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'product': 'projects/sample1/locations/sample2/catalogs/sample3/branches/sample4/products/sample5'}
        mock_args = dict(product='product_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_local_inventories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{product=projects/*/locations/*/catalogs/*/branches/*/products/**}:removeLocalInventories' % client.transport._host, args[1])

def test_remove_local_inventories_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_local_inventories(product_service.RemoveLocalInventoriesRequest(), product='product_value')

def test_remove_local_inventories_rest_error():
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ProductServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ProductServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ProductServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ProductServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ProductServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ProductServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ProductServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ProductServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.ProductServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ProductServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.ProductServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ProductServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ProductServiceGrpcTransport, transports.ProductServiceGrpcAsyncIOTransport, transports.ProductServiceRestTransport])
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
        return 10
    transport = ProductServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ProductServiceGrpcTransport)

def test_product_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ProductServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_product_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.retail_v2.services.product_service.transports.ProductServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ProductServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_product', 'get_product', 'list_products', 'update_product', 'delete_product', 'import_products', 'set_inventory', 'add_fulfillment_places', 'remove_fulfillment_places', 'add_local_inventories', 'remove_local_inventories', 'get_operation', 'list_operations')
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

def test_product_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.retail_v2.services.product_service.transports.ProductServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ProductServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_product_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.retail_v2.services.product_service.transports.ProductServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ProductServiceTransport()
        adc.assert_called_once()

def test_product_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ProductServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ProductServiceGrpcTransport, transports.ProductServiceGrpcAsyncIOTransport])
def test_product_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ProductServiceGrpcTransport, transports.ProductServiceGrpcAsyncIOTransport, transports.ProductServiceRestTransport])
def test_product_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ProductServiceGrpcTransport, grpc_helpers), (transports.ProductServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_product_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ProductServiceGrpcTransport, transports.ProductServiceGrpcAsyncIOTransport])
def test_product_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_product_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ProductServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_product_service_rest_lro_client():
    if False:
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_product_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_product_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_product_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ProductServiceClient(credentials=creds1, transport=transport_name)
    client2 = ProductServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_product._session
    session2 = client2.transport.create_product._session
    assert session1 != session2
    session1 = client1.transport.get_product._session
    session2 = client2.transport.get_product._session
    assert session1 != session2
    session1 = client1.transport.list_products._session
    session2 = client2.transport.list_products._session
    assert session1 != session2
    session1 = client1.transport.update_product._session
    session2 = client2.transport.update_product._session
    assert session1 != session2
    session1 = client1.transport.delete_product._session
    session2 = client2.transport.delete_product._session
    assert session1 != session2
    session1 = client1.transport.import_products._session
    session2 = client2.transport.import_products._session
    assert session1 != session2
    session1 = client1.transport.set_inventory._session
    session2 = client2.transport.set_inventory._session
    assert session1 != session2
    session1 = client1.transport.add_fulfillment_places._session
    session2 = client2.transport.add_fulfillment_places._session
    assert session1 != session2
    session1 = client1.transport.remove_fulfillment_places._session
    session2 = client2.transport.remove_fulfillment_places._session
    assert session1 != session2
    session1 = client1.transport.add_local_inventories._session
    session2 = client2.transport.add_local_inventories._session
    assert session1 != session2
    session1 = client1.transport.remove_local_inventories._session
    session2 = client2.transport.remove_local_inventories._session
    assert session1 != session2

def test_product_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ProductServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_product_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ProductServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ProductServiceGrpcTransport, transports.ProductServiceGrpcAsyncIOTransport])
def test_product_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ProductServiceGrpcTransport, transports.ProductServiceGrpcAsyncIOTransport])
def test_product_service_transport_channel_mtls_with_adc(transport_class):
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

def test_product_service_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_product_service_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_branch_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    catalog = 'whelk'
    branch = 'octopus'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/branches/{branch}'.format(project=project, location=location, catalog=catalog, branch=branch)
    actual = ProductServiceClient.branch_path(project, location, catalog, branch)
    assert expected == actual

def test_parse_branch_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'catalog': 'cuttlefish', 'branch': 'mussel'}
    path = ProductServiceClient.branch_path(**expected)
    actual = ProductServiceClient.parse_branch_path(path)
    assert expected == actual

def test_product_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    catalog = 'scallop'
    branch = 'abalone'
    product = 'squid'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/branches/{branch}/products/{product}'.format(project=project, location=location, catalog=catalog, branch=branch, product=product)
    actual = ProductServiceClient.product_path(project, location, catalog, branch, product)
    assert expected == actual

def test_parse_product_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam', 'location': 'whelk', 'catalog': 'octopus', 'branch': 'oyster', 'product': 'nudibranch'}
    path = ProductServiceClient.product_path(**expected)
    actual = ProductServiceClient.parse_product_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ProductServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = ProductServiceClient.common_billing_account_path(**expected)
    actual = ProductServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ProductServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = ProductServiceClient.common_folder_path(**expected)
    actual = ProductServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ProductServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'abalone'}
    path = ProductServiceClient.common_organization_path(**expected)
    actual = ProductServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = ProductServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam'}
    path = ProductServiceClient.common_project_path(**expected)
    actual = ProductServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ProductServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = ProductServiceClient.common_location_path(**expected)
    actual = ProductServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ProductServiceTransport, '_prep_wrapped_messages') as prep:
        client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ProductServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ProductServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ProductServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = ProductServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ProductServiceClient, transports.ProductServiceGrpcTransport), (ProductServiceAsyncClient, transports.ProductServiceGrpcAsyncIOTransport)])
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