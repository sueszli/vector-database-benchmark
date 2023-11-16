import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
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
from google.protobuf import any_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
from google.type import date_pb2
from google.type import decimal_pb2
from google.type import postal_address_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.channel_v1.services.cloud_channel_service import CloudChannelServiceAsyncClient, CloudChannelServiceClient, pagers, transports
from google.cloud.channel_v1.types import channel_partner_links, common, customers, entitlement_changes, entitlements, offers, operations, products, repricing, service

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
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
    assert CloudChannelServiceClient._get_default_mtls_endpoint(None) is None
    assert CloudChannelServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudChannelServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudChannelServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudChannelServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudChannelServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudChannelServiceClient, 'grpc'), (CloudChannelServiceAsyncClient, 'grpc_asyncio')])
def test_cloud_channel_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'cloudchannel.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudChannelServiceGrpcTransport, 'grpc'), (transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_cloud_channel_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CloudChannelServiceClient, 'grpc'), (CloudChannelServiceAsyncClient, 'grpc_asyncio')])
def test_cloud_channel_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'cloudchannel.googleapis.com:443'

def test_cloud_channel_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CloudChannelServiceClient.get_transport_class()
    available_transports = [transports.CloudChannelServiceGrpcTransport]
    assert transport in available_transports
    transport = CloudChannelServiceClient.get_transport_class('grpc')
    assert transport == transports.CloudChannelServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport, 'grpc'), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(CloudChannelServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudChannelServiceClient))
@mock.patch.object(CloudChannelServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudChannelServiceAsyncClient))
def test_cloud_channel_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(CloudChannelServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudChannelServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport, 'grpc', 'true'), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport, 'grpc', 'false'), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(CloudChannelServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudChannelServiceClient))
@mock.patch.object(CloudChannelServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudChannelServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_channel_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudChannelServiceClient, CloudChannelServiceAsyncClient])
@mock.patch.object(CloudChannelServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudChannelServiceClient))
@mock.patch.object(CloudChannelServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudChannelServiceAsyncClient))
def test_cloud_channel_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport, 'grpc'), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_cloud_channel_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport, 'grpc', grpc_helpers), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_channel_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_channel_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.channel_v1.services.cloud_channel_service.transports.CloudChannelServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudChannelServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport, 'grpc', grpc_helpers), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_channel_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudchannel.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/apps.order',), scopes=None, default_host='cloudchannel.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.ListCustomersRequest, dict])
def test_list_customers(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        call.return_value = service.ListCustomersResponse(next_page_token='next_page_token_value')
        response = client.list_customers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCustomersRequest()
    assert isinstance(response, pagers.ListCustomersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_customers_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        client.list_customers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCustomersRequest()

@pytest.mark.asyncio
async def test_list_customers_async(transport: str='grpc_asyncio', request_type=service.ListCustomersRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCustomersResponse(next_page_token='next_page_token_value'))
        response = await client.list_customers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCustomersRequest()
    assert isinstance(response, pagers.ListCustomersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_customers_async_from_dict():
    await test_list_customers_async(request_type=dict)

def test_list_customers_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCustomersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        call.return_value = service.ListCustomersResponse()
        client.list_customers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_customers_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCustomersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCustomersResponse())
        await client.list_customers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_customers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        call.side_effect = (service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer(), customers.Customer()], next_page_token='abc'), service.ListCustomersResponse(customers=[], next_page_token='def'), service.ListCustomersResponse(customers=[customers.Customer()], next_page_token='ghi'), service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_customers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, customers.Customer) for i in results))

def test_list_customers_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_customers), '__call__') as call:
        call.side_effect = (service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer(), customers.Customer()], next_page_token='abc'), service.ListCustomersResponse(customers=[], next_page_token='def'), service.ListCustomersResponse(customers=[customers.Customer()], next_page_token='ghi'), service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer()]), RuntimeError)
        pages = list(client.list_customers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_customers_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_customers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer(), customers.Customer()], next_page_token='abc'), service.ListCustomersResponse(customers=[], next_page_token='def'), service.ListCustomersResponse(customers=[customers.Customer()], next_page_token='ghi'), service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer()]), RuntimeError)
        async_pager = await client.list_customers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, customers.Customer) for i in responses))

@pytest.mark.asyncio
async def test_list_customers_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_customers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer(), customers.Customer()], next_page_token='abc'), service.ListCustomersResponse(customers=[], next_page_token='def'), service.ListCustomersResponse(customers=[customers.Customer()], next_page_token='ghi'), service.ListCustomersResponse(customers=[customers.Customer(), customers.Customer()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_customers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetCustomerRequest, dict])
def test_get_customer(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        call.return_value = customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value')
        response = client.get_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

def test_get_customer_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        client.get_customer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCustomerRequest()

@pytest.mark.asyncio
async def test_get_customer_async(transport: str='grpc_asyncio', request_type=service.GetCustomerRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value'))
        response = await client.get_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

@pytest.mark.asyncio
async def test_get_customer_async_from_dict():
    await test_get_customer_async(request_type=dict)

def test_get_customer_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCustomerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        call.return_value = customers.Customer()
        client.get_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_customer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCustomerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer())
        await client.get_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_customer_flattened():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        call.return_value = customers.Customer()
        client.get_customer(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_customer_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_customer(service.GetCustomerRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_customer_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_customer), '__call__') as call:
        call.return_value = customers.Customer()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer())
        response = await client.get_customer(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_customer_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_customer(service.GetCustomerRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CheckCloudIdentityAccountsExistRequest, dict])
def test_check_cloud_identity_accounts_exist(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_cloud_identity_accounts_exist), '__call__') as call:
        call.return_value = service.CheckCloudIdentityAccountsExistResponse()
        response = client.check_cloud_identity_accounts_exist(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CheckCloudIdentityAccountsExistRequest()
    assert isinstance(response, service.CheckCloudIdentityAccountsExistResponse)

def test_check_cloud_identity_accounts_exist_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.check_cloud_identity_accounts_exist), '__call__') as call:
        client.check_cloud_identity_accounts_exist()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CheckCloudIdentityAccountsExistRequest()

@pytest.mark.asyncio
async def test_check_cloud_identity_accounts_exist_async(transport: str='grpc_asyncio', request_type=service.CheckCloudIdentityAccountsExistRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_cloud_identity_accounts_exist), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.CheckCloudIdentityAccountsExistResponse())
        response = await client.check_cloud_identity_accounts_exist(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CheckCloudIdentityAccountsExistRequest()
    assert isinstance(response, service.CheckCloudIdentityAccountsExistResponse)

@pytest.mark.asyncio
async def test_check_cloud_identity_accounts_exist_async_from_dict():
    await test_check_cloud_identity_accounts_exist_async(request_type=dict)

def test_check_cloud_identity_accounts_exist_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CheckCloudIdentityAccountsExistRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.check_cloud_identity_accounts_exist), '__call__') as call:
        call.return_value = service.CheckCloudIdentityAccountsExistResponse()
        client.check_cloud_identity_accounts_exist(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_check_cloud_identity_accounts_exist_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CheckCloudIdentityAccountsExistRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.check_cloud_identity_accounts_exist), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.CheckCloudIdentityAccountsExistResponse())
        await client.check_cloud_identity_accounts_exist(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CreateCustomerRequest, dict])
def test_create_customer(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_customer), '__call__') as call:
        call.return_value = customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value')
        response = client.create_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

def test_create_customer_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_customer), '__call__') as call:
        client.create_customer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCustomerRequest()

@pytest.mark.asyncio
async def test_create_customer_async(transport: str='grpc_asyncio', request_type=service.CreateCustomerRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value'))
        response = await client.create_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

@pytest.mark.asyncio
async def test_create_customer_async_from_dict():
    await test_create_customer_async(request_type=dict)

def test_create_customer_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCustomerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_customer), '__call__') as call:
        call.return_value = customers.Customer()
        client.create_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_customer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCustomerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer())
        await client.create_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.UpdateCustomerRequest, dict])
def test_update_customer(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_customer), '__call__') as call:
        call.return_value = customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value')
        response = client.update_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

def test_update_customer_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_customer), '__call__') as call:
        client.update_customer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCustomerRequest()

@pytest.mark.asyncio
async def test_update_customer_async(transport: str='grpc_asyncio', request_type=service.UpdateCustomerRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value'))
        response = await client.update_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

@pytest.mark.asyncio
async def test_update_customer_async_from_dict():
    await test_update_customer_async(request_type=dict)

def test_update_customer_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCustomerRequest()
    request.customer.name = 'name_value'
    with mock.patch.object(type(client.transport.update_customer), '__call__') as call:
        call.return_value = customers.Customer()
        client.update_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_customer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCustomerRequest()
    request.customer.name = 'name_value'
    with mock.patch.object(type(client.transport.update_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer())
        await client.update_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer.name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.DeleteCustomerRequest, dict])
def test_delete_customer(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        call.return_value = None
        response = client.delete_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteCustomerRequest()
    assert response is None

def test_delete_customer_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        client.delete_customer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteCustomerRequest()

@pytest.mark.asyncio
async def test_delete_customer_async(transport: str='grpc_asyncio', request_type=service.DeleteCustomerRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteCustomerRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_customer_async_from_dict():
    await test_delete_customer_async(request_type=dict)

def test_delete_customer_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteCustomerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        call.return_value = None
        client.delete_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_customer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteCustomerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_customer_flattened():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        call.return_value = None
        client.delete_customer(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_customer_flattened_error():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_customer(service.DeleteCustomerRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_customer_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_customer), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_customer(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_customer_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_customer(service.DeleteCustomerRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ImportCustomerRequest, dict])
def test_import_customer(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_customer), '__call__') as call:
        call.return_value = customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value')
        response = client.import_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ImportCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

def test_import_customer_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_customer), '__call__') as call:
        client.import_customer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ImportCustomerRequest()

@pytest.mark.asyncio
async def test_import_customer_async(transport: str='grpc_asyncio', request_type=service.ImportCustomerRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer(name='name_value', org_display_name='org_display_name_value', alternate_email='alternate_email_value', domain='domain_value', cloud_identity_id='cloud_identity_id_value', language_code='language_code_value', channel_partner_id='channel_partner_id_value', correlation_id='correlation_id_value'))
        response = await client.import_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ImportCustomerRequest()
    assert isinstance(response, customers.Customer)
    assert response.name == 'name_value'
    assert response.org_display_name == 'org_display_name_value'
    assert response.alternate_email == 'alternate_email_value'
    assert response.domain == 'domain_value'
    assert response.cloud_identity_id == 'cloud_identity_id_value'
    assert response.language_code == 'language_code_value'
    assert response.channel_partner_id == 'channel_partner_id_value'
    assert response.correlation_id == 'correlation_id_value'

@pytest.mark.asyncio
async def test_import_customer_async_from_dict():
    await test_import_customer_async(request_type=dict)

def test_import_customer_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ImportCustomerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_customer), '__call__') as call:
        call.return_value = customers.Customer()
        client.import_customer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_customer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ImportCustomerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_customer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(customers.Customer())
        await client.import_customer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ProvisionCloudIdentityRequest, dict])
def test_provision_cloud_identity(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.provision_cloud_identity), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.provision_cloud_identity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ProvisionCloudIdentityRequest()
    assert isinstance(response, future.Future)

def test_provision_cloud_identity_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.provision_cloud_identity), '__call__') as call:
        client.provision_cloud_identity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ProvisionCloudIdentityRequest()

@pytest.mark.asyncio
async def test_provision_cloud_identity_async(transport: str='grpc_asyncio', request_type=service.ProvisionCloudIdentityRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.provision_cloud_identity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.provision_cloud_identity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ProvisionCloudIdentityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_provision_cloud_identity_async_from_dict():
    await test_provision_cloud_identity_async(request_type=dict)

def test_provision_cloud_identity_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ProvisionCloudIdentityRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.provision_cloud_identity), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.provision_cloud_identity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

@pytest.mark.asyncio
async def test_provision_cloud_identity_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ProvisionCloudIdentityRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.provision_cloud_identity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.provision_cloud_identity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ListEntitlementsRequest, dict])
def test_list_entitlements(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        call.return_value = service.ListEntitlementsResponse(next_page_token='next_page_token_value')
        response = client.list_entitlements(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntitlementsRequest()
    assert isinstance(response, pagers.ListEntitlementsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entitlements_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        client.list_entitlements()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntitlementsRequest()

@pytest.mark.asyncio
async def test_list_entitlements_async(transport: str='grpc_asyncio', request_type=service.ListEntitlementsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntitlementsResponse(next_page_token='next_page_token_value'))
        response = await client.list_entitlements(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntitlementsRequest()
    assert isinstance(response, pagers.ListEntitlementsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_entitlements_async_from_dict():
    await test_list_entitlements_async(request_type=dict)

def test_list_entitlements_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEntitlementsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        call.return_value = service.ListEntitlementsResponse()
        client.list_entitlements(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_entitlements_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEntitlementsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntitlementsResponse())
        await client.list_entitlements(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_entitlements_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        call.side_effect = (service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement(), entitlements.Entitlement()], next_page_token='abc'), service.ListEntitlementsResponse(entitlements=[], next_page_token='def'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement()], next_page_token='ghi'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_entitlements(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, entitlements.Entitlement) for i in results))

def test_list_entitlements_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entitlements), '__call__') as call:
        call.side_effect = (service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement(), entitlements.Entitlement()], next_page_token='abc'), service.ListEntitlementsResponse(entitlements=[], next_page_token='def'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement()], next_page_token='ghi'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement()]), RuntimeError)
        pages = list(client.list_entitlements(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_entitlements_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entitlements), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement(), entitlements.Entitlement()], next_page_token='abc'), service.ListEntitlementsResponse(entitlements=[], next_page_token='def'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement()], next_page_token='ghi'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement()]), RuntimeError)
        async_pager = await client.list_entitlements(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, entitlements.Entitlement) for i in responses))

@pytest.mark.asyncio
async def test_list_entitlements_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entitlements), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement(), entitlements.Entitlement()], next_page_token='abc'), service.ListEntitlementsResponse(entitlements=[], next_page_token='def'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement()], next_page_token='ghi'), service.ListEntitlementsResponse(entitlements=[entitlements.Entitlement(), entitlements.Entitlement()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_entitlements(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListTransferableSkusRequest, dict])
def test_list_transferable_skus(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        call.return_value = service.ListTransferableSkusResponse(next_page_token='next_page_token_value')
        response = client.list_transferable_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTransferableSkusRequest()
    assert isinstance(response, pagers.ListTransferableSkusPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transferable_skus_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        client.list_transferable_skus()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTransferableSkusRequest()

@pytest.mark.asyncio
async def test_list_transferable_skus_async(transport: str='grpc_asyncio', request_type=service.ListTransferableSkusRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTransferableSkusResponse(next_page_token='next_page_token_value'))
        response = await client.list_transferable_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTransferableSkusRequest()
    assert isinstance(response, pagers.ListTransferableSkusAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transferable_skus_async_from_dict():
    await test_list_transferable_skus_async(request_type=dict)

def test_list_transferable_skus_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListTransferableSkusRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        call.return_value = service.ListTransferableSkusResponse()
        client.list_transferable_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_transferable_skus_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListTransferableSkusRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTransferableSkusResponse())
        await client.list_transferable_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_transferable_skus_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        call.side_effect = (service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku(), entitlements.TransferableSku()], next_page_token='abc'), service.ListTransferableSkusResponse(transferable_skus=[], next_page_token='def'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku()], next_page_token='ghi'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_transferable_skus(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, entitlements.TransferableSku) for i in results))

def test_list_transferable_skus_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__') as call:
        call.side_effect = (service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku(), entitlements.TransferableSku()], next_page_token='abc'), service.ListTransferableSkusResponse(transferable_skus=[], next_page_token='def'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku()], next_page_token='ghi'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku()]), RuntimeError)
        pages = list(client.list_transferable_skus(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transferable_skus_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku(), entitlements.TransferableSku()], next_page_token='abc'), service.ListTransferableSkusResponse(transferable_skus=[], next_page_token='def'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku()], next_page_token='ghi'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku()]), RuntimeError)
        async_pager = await client.list_transferable_skus(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, entitlements.TransferableSku) for i in responses))

@pytest.mark.asyncio
async def test_list_transferable_skus_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transferable_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku(), entitlements.TransferableSku()], next_page_token='abc'), service.ListTransferableSkusResponse(transferable_skus=[], next_page_token='def'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku()], next_page_token='ghi'), service.ListTransferableSkusResponse(transferable_skus=[entitlements.TransferableSku(), entitlements.TransferableSku()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transferable_skus(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListTransferableOffersRequest, dict])
def test_list_transferable_offers(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        call.return_value = service.ListTransferableOffersResponse(next_page_token='next_page_token_value')
        response = client.list_transferable_offers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTransferableOffersRequest()
    assert isinstance(response, pagers.ListTransferableOffersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transferable_offers_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        client.list_transferable_offers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTransferableOffersRequest()

@pytest.mark.asyncio
async def test_list_transferable_offers_async(transport: str='grpc_asyncio', request_type=service.ListTransferableOffersRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTransferableOffersResponse(next_page_token='next_page_token_value'))
        response = await client.list_transferable_offers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTransferableOffersRequest()
    assert isinstance(response, pagers.ListTransferableOffersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transferable_offers_async_from_dict():
    await test_list_transferable_offers_async(request_type=dict)

def test_list_transferable_offers_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListTransferableOffersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        call.return_value = service.ListTransferableOffersResponse()
        client.list_transferable_offers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_transferable_offers_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListTransferableOffersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTransferableOffersResponse())
        await client.list_transferable_offers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_transferable_offers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        call.side_effect = (service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer(), service.TransferableOffer()], next_page_token='abc'), service.ListTransferableOffersResponse(transferable_offers=[], next_page_token='def'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer()], next_page_token='ghi'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_transferable_offers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.TransferableOffer) for i in results))

def test_list_transferable_offers_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__') as call:
        call.side_effect = (service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer(), service.TransferableOffer()], next_page_token='abc'), service.ListTransferableOffersResponse(transferable_offers=[], next_page_token='def'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer()], next_page_token='ghi'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer()]), RuntimeError)
        pages = list(client.list_transferable_offers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transferable_offers_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer(), service.TransferableOffer()], next_page_token='abc'), service.ListTransferableOffersResponse(transferable_offers=[], next_page_token='def'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer()], next_page_token='ghi'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer()]), RuntimeError)
        async_pager = await client.list_transferable_offers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.TransferableOffer) for i in responses))

@pytest.mark.asyncio
async def test_list_transferable_offers_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transferable_offers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer(), service.TransferableOffer()], next_page_token='abc'), service.ListTransferableOffersResponse(transferable_offers=[], next_page_token='def'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer()], next_page_token='ghi'), service.ListTransferableOffersResponse(transferable_offers=[service.TransferableOffer(), service.TransferableOffer()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transferable_offers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetEntitlementRequest, dict])
def test_get_entitlement(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entitlement), '__call__') as call:
        call.return_value = entitlements.Entitlement(name='name_value', offer='offer_value', provisioning_state=entitlements.Entitlement.ProvisioningState.ACTIVE, suspension_reasons=[entitlements.Entitlement.SuspensionReason.RESELLER_INITIATED], purchase_order_id='purchase_order_id_value', billing_account='billing_account_value')
        response = client.get_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEntitlementRequest()
    assert isinstance(response, entitlements.Entitlement)
    assert response.name == 'name_value'
    assert response.offer == 'offer_value'
    assert response.provisioning_state == entitlements.Entitlement.ProvisioningState.ACTIVE
    assert response.suspension_reasons == [entitlements.Entitlement.SuspensionReason.RESELLER_INITIATED]
    assert response.purchase_order_id == 'purchase_order_id_value'
    assert response.billing_account == 'billing_account_value'

def test_get_entitlement_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_entitlement), '__call__') as call:
        client.get_entitlement()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEntitlementRequest()

@pytest.mark.asyncio
async def test_get_entitlement_async(transport: str='grpc_asyncio', request_type=service.GetEntitlementRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entitlements.Entitlement(name='name_value', offer='offer_value', provisioning_state=entitlements.Entitlement.ProvisioningState.ACTIVE, suspension_reasons=[entitlements.Entitlement.SuspensionReason.RESELLER_INITIATED], purchase_order_id='purchase_order_id_value', billing_account='billing_account_value'))
        response = await client.get_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEntitlementRequest()
    assert isinstance(response, entitlements.Entitlement)
    assert response.name == 'name_value'
    assert response.offer == 'offer_value'
    assert response.provisioning_state == entitlements.Entitlement.ProvisioningState.ACTIVE
    assert response.suspension_reasons == [entitlements.Entitlement.SuspensionReason.RESELLER_INITIATED]
    assert response.purchase_order_id == 'purchase_order_id_value'
    assert response.billing_account == 'billing_account_value'

@pytest.mark.asyncio
async def test_get_entitlement_async_from_dict():
    await test_get_entitlement_async(request_type=dict)

def test_get_entitlement_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entitlement), '__call__') as call:
        call.return_value = entitlements.Entitlement()
        client.get_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_entitlement_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entitlements.Entitlement())
        await client.get_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CreateEntitlementRequest, dict])
def test_create_entitlement(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEntitlementRequest()
    assert isinstance(response, future.Future)

def test_create_entitlement_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_entitlement), '__call__') as call:
        client.create_entitlement()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEntitlementRequest()

@pytest.mark.asyncio
async def test_create_entitlement_async(transport: str='grpc_asyncio', request_type=service.CreateEntitlementRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEntitlementRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_entitlement_async_from_dict():
    await test_create_entitlement_async(request_type=dict)

def test_create_entitlement_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateEntitlementRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_entitlement_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateEntitlementRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ChangeParametersRequest, dict])
def test_change_parameters(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.change_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.change_parameters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeParametersRequest()
    assert isinstance(response, future.Future)

def test_change_parameters_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.change_parameters), '__call__') as call:
        client.change_parameters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeParametersRequest()

@pytest.mark.asyncio
async def test_change_parameters_async(transport: str='grpc_asyncio', request_type=service.ChangeParametersRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.change_parameters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.change_parameters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeParametersRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_change_parameters_async_from_dict():
    await test_change_parameters_async(request_type=dict)

def test_change_parameters_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ChangeParametersRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.change_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.change_parameters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_change_parameters_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ChangeParametersRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.change_parameters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.change_parameters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ChangeRenewalSettingsRequest, dict])
def test_change_renewal_settings(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.change_renewal_settings), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.change_renewal_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeRenewalSettingsRequest()
    assert isinstance(response, future.Future)

def test_change_renewal_settings_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.change_renewal_settings), '__call__') as call:
        client.change_renewal_settings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeRenewalSettingsRequest()

@pytest.mark.asyncio
async def test_change_renewal_settings_async(transport: str='grpc_asyncio', request_type=service.ChangeRenewalSettingsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.change_renewal_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.change_renewal_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeRenewalSettingsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_change_renewal_settings_async_from_dict():
    await test_change_renewal_settings_async(request_type=dict)

def test_change_renewal_settings_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ChangeRenewalSettingsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.change_renewal_settings), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.change_renewal_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_change_renewal_settings_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ChangeRenewalSettingsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.change_renewal_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.change_renewal_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ChangeOfferRequest, dict])
def test_change_offer(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.change_offer), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.change_offer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeOfferRequest()
    assert isinstance(response, future.Future)

def test_change_offer_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.change_offer), '__call__') as call:
        client.change_offer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeOfferRequest()

@pytest.mark.asyncio
async def test_change_offer_async(transport: str='grpc_asyncio', request_type=service.ChangeOfferRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.change_offer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.change_offer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ChangeOfferRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_change_offer_async_from_dict():
    await test_change_offer_async(request_type=dict)

def test_change_offer_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ChangeOfferRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.change_offer), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.change_offer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_change_offer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ChangeOfferRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.change_offer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.change_offer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.StartPaidServiceRequest, dict])
def test_start_paid_service(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_paid_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_paid_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StartPaidServiceRequest()
    assert isinstance(response, future.Future)

def test_start_paid_service_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_paid_service), '__call__') as call:
        client.start_paid_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StartPaidServiceRequest()

@pytest.mark.asyncio
async def test_start_paid_service_async(transport: str='grpc_asyncio', request_type=service.StartPaidServiceRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_paid_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_paid_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StartPaidServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_paid_service_async_from_dict():
    await test_start_paid_service_async(request_type=dict)

def test_start_paid_service_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.StartPaidServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_paid_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_paid_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_paid_service_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.StartPaidServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_paid_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_paid_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.SuspendEntitlementRequest, dict])
def test_suspend_entitlement(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.suspend_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.suspend_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SuspendEntitlementRequest()
    assert isinstance(response, future.Future)

def test_suspend_entitlement_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.suspend_entitlement), '__call__') as call:
        client.suspend_entitlement()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SuspendEntitlementRequest()

@pytest.mark.asyncio
async def test_suspend_entitlement_async(transport: str='grpc_asyncio', request_type=service.SuspendEntitlementRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.suspend_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.suspend_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SuspendEntitlementRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_suspend_entitlement_async_from_dict():
    await test_suspend_entitlement_async(request_type=dict)

def test_suspend_entitlement_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SuspendEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.suspend_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.suspend_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_suspend_entitlement_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SuspendEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.suspend_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.suspend_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CancelEntitlementRequest, dict])
def test_cancel_entitlement(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.cancel_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CancelEntitlementRequest()
    assert isinstance(response, future.Future)

def test_cancel_entitlement_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_entitlement), '__call__') as call:
        client.cancel_entitlement()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CancelEntitlementRequest()

@pytest.mark.asyncio
async def test_cancel_entitlement_async(transport: str='grpc_asyncio', request_type=service.CancelEntitlementRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.cancel_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CancelEntitlementRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_cancel_entitlement_async_from_dict():
    await test_cancel_entitlement_async(request_type=dict)

def test_cancel_entitlement_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CancelEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.cancel_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_entitlement_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CancelEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.cancel_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ActivateEntitlementRequest, dict])
def test_activate_entitlement(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.activate_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.activate_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ActivateEntitlementRequest()
    assert isinstance(response, future.Future)

def test_activate_entitlement_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.activate_entitlement), '__call__') as call:
        client.activate_entitlement()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ActivateEntitlementRequest()

@pytest.mark.asyncio
async def test_activate_entitlement_async(transport: str='grpc_asyncio', request_type=service.ActivateEntitlementRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.activate_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.activate_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ActivateEntitlementRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_activate_entitlement_async_from_dict():
    await test_activate_entitlement_async(request_type=dict)

def test_activate_entitlement_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ActivateEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.activate_entitlement), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.activate_entitlement(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_activate_entitlement_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ActivateEntitlementRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.activate_entitlement), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.activate_entitlement(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.TransferEntitlementsRequest, dict])
def test_transfer_entitlements(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.transfer_entitlements), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.transfer_entitlements(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.TransferEntitlementsRequest()
    assert isinstance(response, future.Future)

def test_transfer_entitlements_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.transfer_entitlements), '__call__') as call:
        client.transfer_entitlements()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.TransferEntitlementsRequest()

@pytest.mark.asyncio
async def test_transfer_entitlements_async(transport: str='grpc_asyncio', request_type=service.TransferEntitlementsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.transfer_entitlements), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.transfer_entitlements(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.TransferEntitlementsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_transfer_entitlements_async_from_dict():
    await test_transfer_entitlements_async(request_type=dict)

def test_transfer_entitlements_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.TransferEntitlementsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.transfer_entitlements), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.transfer_entitlements(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_transfer_entitlements_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.TransferEntitlementsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.transfer_entitlements), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.transfer_entitlements(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.TransferEntitlementsToGoogleRequest, dict])
def test_transfer_entitlements_to_google(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.transfer_entitlements_to_google), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.transfer_entitlements_to_google(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.TransferEntitlementsToGoogleRequest()
    assert isinstance(response, future.Future)

def test_transfer_entitlements_to_google_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.transfer_entitlements_to_google), '__call__') as call:
        client.transfer_entitlements_to_google()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.TransferEntitlementsToGoogleRequest()

@pytest.mark.asyncio
async def test_transfer_entitlements_to_google_async(transport: str='grpc_asyncio', request_type=service.TransferEntitlementsToGoogleRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.transfer_entitlements_to_google), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.transfer_entitlements_to_google(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.TransferEntitlementsToGoogleRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_transfer_entitlements_to_google_async_from_dict():
    await test_transfer_entitlements_to_google_async(request_type=dict)

def test_transfer_entitlements_to_google_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.TransferEntitlementsToGoogleRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.transfer_entitlements_to_google), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.transfer_entitlements_to_google(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_transfer_entitlements_to_google_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.TransferEntitlementsToGoogleRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.transfer_entitlements_to_google), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.transfer_entitlements_to_google(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ListChannelPartnerLinksRequest, dict])
def test_list_channel_partner_links(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        call.return_value = service.ListChannelPartnerLinksResponse(next_page_token='next_page_token_value')
        response = client.list_channel_partner_links(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelPartnerLinksRequest()
    assert isinstance(response, pagers.ListChannelPartnerLinksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_channel_partner_links_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        client.list_channel_partner_links()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelPartnerLinksRequest()

@pytest.mark.asyncio
async def test_list_channel_partner_links_async(transport: str='grpc_asyncio', request_type=service.ListChannelPartnerLinksRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelPartnerLinksResponse(next_page_token='next_page_token_value'))
        response = await client.list_channel_partner_links(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelPartnerLinksRequest()
    assert isinstance(response, pagers.ListChannelPartnerLinksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_channel_partner_links_async_from_dict():
    await test_list_channel_partner_links_async(request_type=dict)

def test_list_channel_partner_links_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListChannelPartnerLinksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        call.return_value = service.ListChannelPartnerLinksResponse()
        client.list_channel_partner_links(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_channel_partner_links_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListChannelPartnerLinksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelPartnerLinksResponse())
        await client.list_channel_partner_links(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_channel_partner_links_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        call.side_effect = (service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()], next_page_token='abc'), service.ListChannelPartnerLinksResponse(channel_partner_links=[], next_page_token='def'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink()], next_page_token='ghi'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_channel_partner_links(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, channel_partner_links.ChannelPartnerLink) for i in results))

def test_list_channel_partner_links_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__') as call:
        call.side_effect = (service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()], next_page_token='abc'), service.ListChannelPartnerLinksResponse(channel_partner_links=[], next_page_token='def'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink()], next_page_token='ghi'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()]), RuntimeError)
        pages = list(client.list_channel_partner_links(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_channel_partner_links_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()], next_page_token='abc'), service.ListChannelPartnerLinksResponse(channel_partner_links=[], next_page_token='def'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink()], next_page_token='ghi'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()]), RuntimeError)
        async_pager = await client.list_channel_partner_links(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, channel_partner_links.ChannelPartnerLink) for i in responses))

@pytest.mark.asyncio
async def test_list_channel_partner_links_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channel_partner_links), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()], next_page_token='abc'), service.ListChannelPartnerLinksResponse(channel_partner_links=[], next_page_token='def'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink()], next_page_token='ghi'), service.ListChannelPartnerLinksResponse(channel_partner_links=[channel_partner_links.ChannelPartnerLink(), channel_partner_links.ChannelPartnerLink()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_channel_partner_links(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetChannelPartnerLinkRequest, dict])
def test_get_channel_partner_link(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel_partner_link), '__call__') as call:
        call.return_value = channel_partner_links.ChannelPartnerLink(name='name_value', reseller_cloud_identity_id='reseller_cloud_identity_id_value', link_state=channel_partner_links.ChannelPartnerLinkState.INVITED, invite_link_uri='invite_link_uri_value', public_id='public_id_value')
        response = client.get_channel_partner_link(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelPartnerLinkRequest()
    assert isinstance(response, channel_partner_links.ChannelPartnerLink)
    assert response.name == 'name_value'
    assert response.reseller_cloud_identity_id == 'reseller_cloud_identity_id_value'
    assert response.link_state == channel_partner_links.ChannelPartnerLinkState.INVITED
    assert response.invite_link_uri == 'invite_link_uri_value'
    assert response.public_id == 'public_id_value'

def test_get_channel_partner_link_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_channel_partner_link), '__call__') as call:
        client.get_channel_partner_link()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelPartnerLinkRequest()

@pytest.mark.asyncio
async def test_get_channel_partner_link_async(transport: str='grpc_asyncio', request_type=service.GetChannelPartnerLinkRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel_partner_link), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_partner_links.ChannelPartnerLink(name='name_value', reseller_cloud_identity_id='reseller_cloud_identity_id_value', link_state=channel_partner_links.ChannelPartnerLinkState.INVITED, invite_link_uri='invite_link_uri_value', public_id='public_id_value'))
        response = await client.get_channel_partner_link(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelPartnerLinkRequest()
    assert isinstance(response, channel_partner_links.ChannelPartnerLink)
    assert response.name == 'name_value'
    assert response.reseller_cloud_identity_id == 'reseller_cloud_identity_id_value'
    assert response.link_state == channel_partner_links.ChannelPartnerLinkState.INVITED
    assert response.invite_link_uri == 'invite_link_uri_value'
    assert response.public_id == 'public_id_value'

@pytest.mark.asyncio
async def test_get_channel_partner_link_async_from_dict():
    await test_get_channel_partner_link_async(request_type=dict)

def test_get_channel_partner_link_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetChannelPartnerLinkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel_partner_link), '__call__') as call:
        call.return_value = channel_partner_links.ChannelPartnerLink()
        client.get_channel_partner_link(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_channel_partner_link_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetChannelPartnerLinkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel_partner_link), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_partner_links.ChannelPartnerLink())
        await client.get_channel_partner_link(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CreateChannelPartnerLinkRequest, dict])
def test_create_channel_partner_link(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_partner_link), '__call__') as call:
        call.return_value = channel_partner_links.ChannelPartnerLink(name='name_value', reseller_cloud_identity_id='reseller_cloud_identity_id_value', link_state=channel_partner_links.ChannelPartnerLinkState.INVITED, invite_link_uri='invite_link_uri_value', public_id='public_id_value')
        response = client.create_channel_partner_link(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelPartnerLinkRequest()
    assert isinstance(response, channel_partner_links.ChannelPartnerLink)
    assert response.name == 'name_value'
    assert response.reseller_cloud_identity_id == 'reseller_cloud_identity_id_value'
    assert response.link_state == channel_partner_links.ChannelPartnerLinkState.INVITED
    assert response.invite_link_uri == 'invite_link_uri_value'
    assert response.public_id == 'public_id_value'

def test_create_channel_partner_link_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_channel_partner_link), '__call__') as call:
        client.create_channel_partner_link()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelPartnerLinkRequest()

@pytest.mark.asyncio
async def test_create_channel_partner_link_async(transport: str='grpc_asyncio', request_type=service.CreateChannelPartnerLinkRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_partner_link), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_partner_links.ChannelPartnerLink(name='name_value', reseller_cloud_identity_id='reseller_cloud_identity_id_value', link_state=channel_partner_links.ChannelPartnerLinkState.INVITED, invite_link_uri='invite_link_uri_value', public_id='public_id_value'))
        response = await client.create_channel_partner_link(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelPartnerLinkRequest()
    assert isinstance(response, channel_partner_links.ChannelPartnerLink)
    assert response.name == 'name_value'
    assert response.reseller_cloud_identity_id == 'reseller_cloud_identity_id_value'
    assert response.link_state == channel_partner_links.ChannelPartnerLinkState.INVITED
    assert response.invite_link_uri == 'invite_link_uri_value'
    assert response.public_id == 'public_id_value'

@pytest.mark.asyncio
async def test_create_channel_partner_link_async_from_dict():
    await test_create_channel_partner_link_async(request_type=dict)

def test_create_channel_partner_link_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateChannelPartnerLinkRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_partner_link), '__call__') as call:
        call.return_value = channel_partner_links.ChannelPartnerLink()
        client.create_channel_partner_link(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_channel_partner_link_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateChannelPartnerLinkRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_partner_link), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_partner_links.ChannelPartnerLink())
        await client.create_channel_partner_link(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.UpdateChannelPartnerLinkRequest, dict])
def test_update_channel_partner_link(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel_partner_link), '__call__') as call:
        call.return_value = channel_partner_links.ChannelPartnerLink(name='name_value', reseller_cloud_identity_id='reseller_cloud_identity_id_value', link_state=channel_partner_links.ChannelPartnerLinkState.INVITED, invite_link_uri='invite_link_uri_value', public_id='public_id_value')
        response = client.update_channel_partner_link(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelPartnerLinkRequest()
    assert isinstance(response, channel_partner_links.ChannelPartnerLink)
    assert response.name == 'name_value'
    assert response.reseller_cloud_identity_id == 'reseller_cloud_identity_id_value'
    assert response.link_state == channel_partner_links.ChannelPartnerLinkState.INVITED
    assert response.invite_link_uri == 'invite_link_uri_value'
    assert response.public_id == 'public_id_value'

def test_update_channel_partner_link_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_channel_partner_link), '__call__') as call:
        client.update_channel_partner_link()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelPartnerLinkRequest()

@pytest.mark.asyncio
async def test_update_channel_partner_link_async(transport: str='grpc_asyncio', request_type=service.UpdateChannelPartnerLinkRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel_partner_link), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_partner_links.ChannelPartnerLink(name='name_value', reseller_cloud_identity_id='reseller_cloud_identity_id_value', link_state=channel_partner_links.ChannelPartnerLinkState.INVITED, invite_link_uri='invite_link_uri_value', public_id='public_id_value'))
        response = await client.update_channel_partner_link(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelPartnerLinkRequest()
    assert isinstance(response, channel_partner_links.ChannelPartnerLink)
    assert response.name == 'name_value'
    assert response.reseller_cloud_identity_id == 'reseller_cloud_identity_id_value'
    assert response.link_state == channel_partner_links.ChannelPartnerLinkState.INVITED
    assert response.invite_link_uri == 'invite_link_uri_value'
    assert response.public_id == 'public_id_value'

@pytest.mark.asyncio
async def test_update_channel_partner_link_async_from_dict():
    await test_update_channel_partner_link_async(request_type=dict)

def test_update_channel_partner_link_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateChannelPartnerLinkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel_partner_link), '__call__') as call:
        call.return_value = channel_partner_links.ChannelPartnerLink()
        client.update_channel_partner_link(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_channel_partner_link_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateChannelPartnerLinkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel_partner_link), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_partner_links.ChannelPartnerLink())
        await client.update_channel_partner_link(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.GetCustomerRepricingConfigRequest, dict])
def test_get_customer_repricing_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig(name='name_value')
        response = client.get_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCustomerRepricingConfigRequest()
    assert isinstance(response, repricing.CustomerRepricingConfig)
    assert response.name == 'name_value'

def test_get_customer_repricing_config_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        client.get_customer_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCustomerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_get_customer_repricing_config_async(transport: str='grpc_asyncio', request_type=service.GetCustomerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig(name='name_value'))
        response = await client.get_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCustomerRepricingConfigRequest()
    assert isinstance(response, repricing.CustomerRepricingConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_customer_repricing_config_async_from_dict():
    await test_get_customer_repricing_config_async(request_type=dict)

def test_get_customer_repricing_config_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCustomerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        client.get_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_customer_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCustomerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig())
        await client.get_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_customer_repricing_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        client.get_customer_repricing_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_customer_repricing_config_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_customer_repricing_config(service.GetCustomerRepricingConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_customer_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig())
        response = await client.get_customer_repricing_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_customer_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_customer_repricing_config(service.GetCustomerRepricingConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListCustomerRepricingConfigsRequest, dict])
def test_list_customer_repricing_configs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.return_value = service.ListCustomerRepricingConfigsResponse(next_page_token='next_page_token_value')
        response = client.list_customer_repricing_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCustomerRepricingConfigsRequest()
    assert isinstance(response, pagers.ListCustomerRepricingConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_customer_repricing_configs_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        client.list_customer_repricing_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCustomerRepricingConfigsRequest()

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_async(transport: str='grpc_asyncio', request_type=service.ListCustomerRepricingConfigsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCustomerRepricingConfigsResponse(next_page_token='next_page_token_value'))
        response = await client.list_customer_repricing_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCustomerRepricingConfigsRequest()
    assert isinstance(response, pagers.ListCustomerRepricingConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_async_from_dict():
    await test_list_customer_repricing_configs_async(request_type=dict)

def test_list_customer_repricing_configs_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCustomerRepricingConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.return_value = service.ListCustomerRepricingConfigsResponse()
        client.list_customer_repricing_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCustomerRepricingConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCustomerRepricingConfigsResponse())
        await client.list_customer_repricing_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_customer_repricing_configs_flattened():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.return_value = service.ListCustomerRepricingConfigsResponse()
        client.list_customer_repricing_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_customer_repricing_configs_flattened_error():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_customer_repricing_configs(service.ListCustomerRepricingConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.return_value = service.ListCustomerRepricingConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCustomerRepricingConfigsResponse())
        response = await client.list_customer_repricing_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_customer_repricing_configs(service.ListCustomerRepricingConfigsRequest(), parent='parent_value')

def test_list_customer_repricing_configs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.side_effect = (service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()], next_page_token='abc'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[], next_page_token='def'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig()], next_page_token='ghi'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_customer_repricing_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repricing.CustomerRepricingConfig) for i in results))

def test_list_customer_repricing_configs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__') as call:
        call.side_effect = (service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()], next_page_token='abc'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[], next_page_token='def'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig()], next_page_token='ghi'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()]), RuntimeError)
        pages = list(client.list_customer_repricing_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()], next_page_token='abc'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[], next_page_token='def'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig()], next_page_token='ghi'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()]), RuntimeError)
        async_pager = await client.list_customer_repricing_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, repricing.CustomerRepricingConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_customer_repricing_configs_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_customer_repricing_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()], next_page_token='abc'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[], next_page_token='def'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig()], next_page_token='ghi'), service.ListCustomerRepricingConfigsResponse(customer_repricing_configs=[repricing.CustomerRepricingConfig(), repricing.CustomerRepricingConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_customer_repricing_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateCustomerRepricingConfigRequest, dict])
def test_create_customer_repricing_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig(name='name_value')
        response = client.create_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCustomerRepricingConfigRequest()
    assert isinstance(response, repricing.CustomerRepricingConfig)
    assert response.name == 'name_value'

def test_create_customer_repricing_config_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        client.create_customer_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCustomerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_create_customer_repricing_config_async(transport: str='grpc_asyncio', request_type=service.CreateCustomerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig(name='name_value'))
        response = await client.create_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCustomerRepricingConfigRequest()
    assert isinstance(response, repricing.CustomerRepricingConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_customer_repricing_config_async_from_dict():
    await test_create_customer_repricing_config_async(request_type=dict)

def test_create_customer_repricing_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCustomerRepricingConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        client.create_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_customer_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCustomerRepricingConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig())
        await client.create_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_customer_repricing_config_flattened():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        client.create_customer_repricing_config(parent='parent_value', customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].customer_repricing_config
        mock_val = repricing.CustomerRepricingConfig(name='name_value')
        assert arg == mock_val

def test_create_customer_repricing_config_flattened_error():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_customer_repricing_config(service.CreateCustomerRepricingConfigRequest(), parent='parent_value', customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))

@pytest.mark.asyncio
async def test_create_customer_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig())
        response = await client.create_customer_repricing_config(parent='parent_value', customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].customer_repricing_config
        mock_val = repricing.CustomerRepricingConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_customer_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_customer_repricing_config(service.CreateCustomerRepricingConfigRequest(), parent='parent_value', customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [service.UpdateCustomerRepricingConfigRequest, dict])
def test_update_customer_repricing_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig(name='name_value')
        response = client.update_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCustomerRepricingConfigRequest()
    assert isinstance(response, repricing.CustomerRepricingConfig)
    assert response.name == 'name_value'

def test_update_customer_repricing_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        client.update_customer_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCustomerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_update_customer_repricing_config_async(transport: str='grpc_asyncio', request_type=service.UpdateCustomerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig(name='name_value'))
        response = await client.update_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCustomerRepricingConfigRequest()
    assert isinstance(response, repricing.CustomerRepricingConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_customer_repricing_config_async_from_dict():
    await test_update_customer_repricing_config_async(request_type=dict)

def test_update_customer_repricing_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCustomerRepricingConfigRequest()
    request.customer_repricing_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        client.update_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer_repricing_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_customer_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCustomerRepricingConfigRequest()
    request.customer_repricing_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig())
        await client.update_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer_repricing_config.name=name_value') in kw['metadata']

def test_update_customer_repricing_config_flattened():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        client.update_customer_repricing_config(customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].customer_repricing_config
        mock_val = repricing.CustomerRepricingConfig(name='name_value')
        assert arg == mock_val

def test_update_customer_repricing_config_flattened_error():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_customer_repricing_config(service.UpdateCustomerRepricingConfigRequest(), customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))

@pytest.mark.asyncio
async def test_update_customer_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_customer_repricing_config), '__call__') as call:
        call.return_value = repricing.CustomerRepricingConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.CustomerRepricingConfig())
        response = await client.update_customer_repricing_config(customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].customer_repricing_config
        mock_val = repricing.CustomerRepricingConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_customer_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_customer_repricing_config(service.UpdateCustomerRepricingConfigRequest(), customer_repricing_config=repricing.CustomerRepricingConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [service.DeleteCustomerRepricingConfigRequest, dict])
def test_delete_customer_repricing_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        call.return_value = None
        response = client.delete_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteCustomerRepricingConfigRequest()
    assert response is None

def test_delete_customer_repricing_config_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        client.delete_customer_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteCustomerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_delete_customer_repricing_config_async(transport: str='grpc_asyncio', request_type=service.DeleteCustomerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteCustomerRepricingConfigRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_customer_repricing_config_async_from_dict():
    await test_delete_customer_repricing_config_async(request_type=dict)

def test_delete_customer_repricing_config_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteCustomerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        call.return_value = None
        client.delete_customer_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_customer_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteCustomerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_customer_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_customer_repricing_config_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        call.return_value = None
        client.delete_customer_repricing_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_customer_repricing_config_flattened_error():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_customer_repricing_config(service.DeleteCustomerRepricingConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_customer_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_customer_repricing_config), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_customer_repricing_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_customer_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_customer_repricing_config(service.DeleteCustomerRepricingConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetChannelPartnerRepricingConfigRequest, dict])
def test_get_channel_partner_repricing_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig(name='name_value')
        response = client.get_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelPartnerRepricingConfigRequest()
    assert isinstance(response, repricing.ChannelPartnerRepricingConfig)
    assert response.name == 'name_value'

def test_get_channel_partner_repricing_config_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        client.get_channel_partner_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelPartnerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_get_channel_partner_repricing_config_async(transport: str='grpc_asyncio', request_type=service.GetChannelPartnerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig(name='name_value'))
        response = await client.get_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelPartnerRepricingConfigRequest()
    assert isinstance(response, repricing.ChannelPartnerRepricingConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_channel_partner_repricing_config_async_from_dict():
    await test_get_channel_partner_repricing_config_async(request_type=dict)

def test_get_channel_partner_repricing_config_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetChannelPartnerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        client.get_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_channel_partner_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetChannelPartnerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig())
        await client.get_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_channel_partner_repricing_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        client.get_channel_partner_repricing_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_channel_partner_repricing_config_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_channel_partner_repricing_config(service.GetChannelPartnerRepricingConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_channel_partner_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig())
        response = await client.get_channel_partner_repricing_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_channel_partner_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_channel_partner_repricing_config(service.GetChannelPartnerRepricingConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListChannelPartnerRepricingConfigsRequest, dict])
def test_list_channel_partner_repricing_configs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.return_value = service.ListChannelPartnerRepricingConfigsResponse(next_page_token='next_page_token_value')
        response = client.list_channel_partner_repricing_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelPartnerRepricingConfigsRequest()
    assert isinstance(response, pagers.ListChannelPartnerRepricingConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_channel_partner_repricing_configs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        client.list_channel_partner_repricing_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelPartnerRepricingConfigsRequest()

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_async(transport: str='grpc_asyncio', request_type=service.ListChannelPartnerRepricingConfigsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelPartnerRepricingConfigsResponse(next_page_token='next_page_token_value'))
        response = await client.list_channel_partner_repricing_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelPartnerRepricingConfigsRequest()
    assert isinstance(response, pagers.ListChannelPartnerRepricingConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_async_from_dict():
    await test_list_channel_partner_repricing_configs_async(request_type=dict)

def test_list_channel_partner_repricing_configs_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListChannelPartnerRepricingConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.return_value = service.ListChannelPartnerRepricingConfigsResponse()
        client.list_channel_partner_repricing_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListChannelPartnerRepricingConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelPartnerRepricingConfigsResponse())
        await client.list_channel_partner_repricing_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_channel_partner_repricing_configs_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.return_value = service.ListChannelPartnerRepricingConfigsResponse()
        client.list_channel_partner_repricing_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_channel_partner_repricing_configs_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_channel_partner_repricing_configs(service.ListChannelPartnerRepricingConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.return_value = service.ListChannelPartnerRepricingConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelPartnerRepricingConfigsResponse())
        response = await client.list_channel_partner_repricing_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_channel_partner_repricing_configs(service.ListChannelPartnerRepricingConfigsRequest(), parent='parent_value')

def test_list_channel_partner_repricing_configs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.side_effect = (service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()], next_page_token='abc'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[], next_page_token='def'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig()], next_page_token='ghi'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_channel_partner_repricing_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repricing.ChannelPartnerRepricingConfig) for i in results))

def test_list_channel_partner_repricing_configs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__') as call:
        call.side_effect = (service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()], next_page_token='abc'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[], next_page_token='def'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig()], next_page_token='ghi'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()]), RuntimeError)
        pages = list(client.list_channel_partner_repricing_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()], next_page_token='abc'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[], next_page_token='def'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig()], next_page_token='ghi'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()]), RuntimeError)
        async_pager = await client.list_channel_partner_repricing_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, repricing.ChannelPartnerRepricingConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_channel_partner_repricing_configs_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channel_partner_repricing_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()], next_page_token='abc'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[], next_page_token='def'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig()], next_page_token='ghi'), service.ListChannelPartnerRepricingConfigsResponse(channel_partner_repricing_configs=[repricing.ChannelPartnerRepricingConfig(), repricing.ChannelPartnerRepricingConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_channel_partner_repricing_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateChannelPartnerRepricingConfigRequest, dict])
def test_create_channel_partner_repricing_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig(name='name_value')
        response = client.create_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelPartnerRepricingConfigRequest()
    assert isinstance(response, repricing.ChannelPartnerRepricingConfig)
    assert response.name == 'name_value'

def test_create_channel_partner_repricing_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        client.create_channel_partner_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelPartnerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_create_channel_partner_repricing_config_async(transport: str='grpc_asyncio', request_type=service.CreateChannelPartnerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig(name='name_value'))
        response = await client.create_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelPartnerRepricingConfigRequest()
    assert isinstance(response, repricing.ChannelPartnerRepricingConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_channel_partner_repricing_config_async_from_dict():
    await test_create_channel_partner_repricing_config_async(request_type=dict)

def test_create_channel_partner_repricing_config_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateChannelPartnerRepricingConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        client.create_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_channel_partner_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateChannelPartnerRepricingConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig())
        await client.create_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_channel_partner_repricing_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        client.create_channel_partner_repricing_config(parent='parent_value', channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel_partner_repricing_config
        mock_val = repricing.ChannelPartnerRepricingConfig(name='name_value')
        assert arg == mock_val

def test_create_channel_partner_repricing_config_flattened_error():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_channel_partner_repricing_config(service.CreateChannelPartnerRepricingConfigRequest(), parent='parent_value', channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))

@pytest.mark.asyncio
async def test_create_channel_partner_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig())
        response = await client.create_channel_partner_repricing_config(parent='parent_value', channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel_partner_repricing_config
        mock_val = repricing.ChannelPartnerRepricingConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_channel_partner_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_channel_partner_repricing_config(service.CreateChannelPartnerRepricingConfigRequest(), parent='parent_value', channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [service.UpdateChannelPartnerRepricingConfigRequest, dict])
def test_update_channel_partner_repricing_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig(name='name_value')
        response = client.update_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelPartnerRepricingConfigRequest()
    assert isinstance(response, repricing.ChannelPartnerRepricingConfig)
    assert response.name == 'name_value'

def test_update_channel_partner_repricing_config_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        client.update_channel_partner_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelPartnerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_update_channel_partner_repricing_config_async(transport: str='grpc_asyncio', request_type=service.UpdateChannelPartnerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig(name='name_value'))
        response = await client.update_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelPartnerRepricingConfigRequest()
    assert isinstance(response, repricing.ChannelPartnerRepricingConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_channel_partner_repricing_config_async_from_dict():
    await test_update_channel_partner_repricing_config_async(request_type=dict)

def test_update_channel_partner_repricing_config_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateChannelPartnerRepricingConfigRequest()
    request.channel_partner_repricing_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        client.update_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'channel_partner_repricing_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_channel_partner_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateChannelPartnerRepricingConfigRequest()
    request.channel_partner_repricing_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig())
        await client.update_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'channel_partner_repricing_config.name=name_value') in kw['metadata']

def test_update_channel_partner_repricing_config_flattened():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        client.update_channel_partner_repricing_config(channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].channel_partner_repricing_config
        mock_val = repricing.ChannelPartnerRepricingConfig(name='name_value')
        assert arg == mock_val

def test_update_channel_partner_repricing_config_flattened_error():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_channel_partner_repricing_config(service.UpdateChannelPartnerRepricingConfigRequest(), channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))

@pytest.mark.asyncio
async def test_update_channel_partner_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_channel_partner_repricing_config), '__call__') as call:
        call.return_value = repricing.ChannelPartnerRepricingConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repricing.ChannelPartnerRepricingConfig())
        response = await client.update_channel_partner_repricing_config(channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].channel_partner_repricing_config
        mock_val = repricing.ChannelPartnerRepricingConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_channel_partner_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_channel_partner_repricing_config(service.UpdateChannelPartnerRepricingConfigRequest(), channel_partner_repricing_config=repricing.ChannelPartnerRepricingConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [service.DeleteChannelPartnerRepricingConfigRequest, dict])
def test_delete_channel_partner_repricing_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        call.return_value = None
        response = client.delete_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteChannelPartnerRepricingConfigRequest()
    assert response is None

def test_delete_channel_partner_repricing_config_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        client.delete_channel_partner_repricing_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteChannelPartnerRepricingConfigRequest()

@pytest.mark.asyncio
async def test_delete_channel_partner_repricing_config_async(transport: str='grpc_asyncio', request_type=service.DeleteChannelPartnerRepricingConfigRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteChannelPartnerRepricingConfigRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_channel_partner_repricing_config_async_from_dict():
    await test_delete_channel_partner_repricing_config_async(request_type=dict)

def test_delete_channel_partner_repricing_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteChannelPartnerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        call.return_value = None
        client.delete_channel_partner_repricing_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_channel_partner_repricing_config_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteChannelPartnerRepricingConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_channel_partner_repricing_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_channel_partner_repricing_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        call.return_value = None
        client.delete_channel_partner_repricing_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_channel_partner_repricing_config_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_channel_partner_repricing_config(service.DeleteChannelPartnerRepricingConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_channel_partner_repricing_config_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel_partner_repricing_config), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_channel_partner_repricing_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_channel_partner_repricing_config_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_channel_partner_repricing_config(service.DeleteChannelPartnerRepricingConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListSkuGroupsRequest, dict])
def test_list_sku_groups(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.return_value = service.ListSkuGroupsResponse(next_page_token='next_page_token_value')
        response = client.list_sku_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkuGroupsRequest()
    assert isinstance(response, pagers.ListSkuGroupsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_sku_groups_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        client.list_sku_groups()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkuGroupsRequest()

@pytest.mark.asyncio
async def test_list_sku_groups_async(transport: str='grpc_asyncio', request_type=service.ListSkuGroupsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkuGroupsResponse(next_page_token='next_page_token_value'))
        response = await client.list_sku_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkuGroupsRequest()
    assert isinstance(response, pagers.ListSkuGroupsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_sku_groups_async_from_dict():
    await test_list_sku_groups_async(request_type=dict)

def test_list_sku_groups_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSkuGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.return_value = service.ListSkuGroupsResponse()
        client.list_sku_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_sku_groups_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSkuGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkuGroupsResponse())
        await client.list_sku_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_sku_groups_flattened():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.return_value = service.ListSkuGroupsResponse()
        client.list_sku_groups(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_sku_groups_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_sku_groups(service.ListSkuGroupsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_sku_groups_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.return_value = service.ListSkuGroupsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkuGroupsResponse())
        response = await client.list_sku_groups(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_sku_groups_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_sku_groups(service.ListSkuGroupsRequest(), parent='parent_value')

def test_list_sku_groups_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.side_effect = (service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup(), service.SkuGroup()], next_page_token='abc'), service.ListSkuGroupsResponse(sku_groups=[], next_page_token='def'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup()], next_page_token='ghi'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_sku_groups(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.SkuGroup) for i in results))

def test_list_sku_groups_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__') as call:
        call.side_effect = (service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup(), service.SkuGroup()], next_page_token='abc'), service.ListSkuGroupsResponse(sku_groups=[], next_page_token='def'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup()], next_page_token='ghi'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup()]), RuntimeError)
        pages = list(client.list_sku_groups(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_sku_groups_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup(), service.SkuGroup()], next_page_token='abc'), service.ListSkuGroupsResponse(sku_groups=[], next_page_token='def'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup()], next_page_token='ghi'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup()]), RuntimeError)
        async_pager = await client.list_sku_groups(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.SkuGroup) for i in responses))

@pytest.mark.asyncio
async def test_list_sku_groups_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_sku_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup(), service.SkuGroup()], next_page_token='abc'), service.ListSkuGroupsResponse(sku_groups=[], next_page_token='def'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup()], next_page_token='ghi'), service.ListSkuGroupsResponse(sku_groups=[service.SkuGroup(), service.SkuGroup()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_sku_groups(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListSkuGroupBillableSkusRequest, dict])
def test_list_sku_group_billable_skus(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.return_value = service.ListSkuGroupBillableSkusResponse(next_page_token='next_page_token_value')
        response = client.list_sku_group_billable_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkuGroupBillableSkusRequest()
    assert isinstance(response, pagers.ListSkuGroupBillableSkusPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_sku_group_billable_skus_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        client.list_sku_group_billable_skus()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkuGroupBillableSkusRequest()

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_async(transport: str='grpc_asyncio', request_type=service.ListSkuGroupBillableSkusRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkuGroupBillableSkusResponse(next_page_token='next_page_token_value'))
        response = await client.list_sku_group_billable_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkuGroupBillableSkusRequest()
    assert isinstance(response, pagers.ListSkuGroupBillableSkusAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_async_from_dict():
    await test_list_sku_group_billable_skus_async(request_type=dict)

def test_list_sku_group_billable_skus_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSkuGroupBillableSkusRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.return_value = service.ListSkuGroupBillableSkusResponse()
        client.list_sku_group_billable_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSkuGroupBillableSkusRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkuGroupBillableSkusResponse())
        await client.list_sku_group_billable_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_sku_group_billable_skus_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.return_value = service.ListSkuGroupBillableSkusResponse()
        client.list_sku_group_billable_skus(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_sku_group_billable_skus_flattened_error():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_sku_group_billable_skus(service.ListSkuGroupBillableSkusRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.return_value = service.ListSkuGroupBillableSkusResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkuGroupBillableSkusResponse())
        response = await client.list_sku_group_billable_skus(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_sku_group_billable_skus(service.ListSkuGroupBillableSkusRequest(), parent='parent_value')

def test_list_sku_group_billable_skus_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.side_effect = (service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku(), service.BillableSku()], next_page_token='abc'), service.ListSkuGroupBillableSkusResponse(billable_skus=[], next_page_token='def'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku()], next_page_token='ghi'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_sku_group_billable_skus(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.BillableSku) for i in results))

def test_list_sku_group_billable_skus_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__') as call:
        call.side_effect = (service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku(), service.BillableSku()], next_page_token='abc'), service.ListSkuGroupBillableSkusResponse(billable_skus=[], next_page_token='def'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku()], next_page_token='ghi'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku()]), RuntimeError)
        pages = list(client.list_sku_group_billable_skus(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku(), service.BillableSku()], next_page_token='abc'), service.ListSkuGroupBillableSkusResponse(billable_skus=[], next_page_token='def'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku()], next_page_token='ghi'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku()]), RuntimeError)
        async_pager = await client.list_sku_group_billable_skus(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.BillableSku) for i in responses))

@pytest.mark.asyncio
async def test_list_sku_group_billable_skus_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_sku_group_billable_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku(), service.BillableSku()], next_page_token='abc'), service.ListSkuGroupBillableSkusResponse(billable_skus=[], next_page_token='def'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku()], next_page_token='ghi'), service.ListSkuGroupBillableSkusResponse(billable_skus=[service.BillableSku(), service.BillableSku()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_sku_group_billable_skus(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.LookupOfferRequest, dict])
def test_lookup_offer(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.lookup_offer), '__call__') as call:
        call.return_value = offers.Offer(name='name_value', deal_code='deal_code_value')
        response = client.lookup_offer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupOfferRequest()
    assert isinstance(response, offers.Offer)
    assert response.name == 'name_value'
    assert response.deal_code == 'deal_code_value'

def test_lookup_offer_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.lookup_offer), '__call__') as call:
        client.lookup_offer()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupOfferRequest()

@pytest.mark.asyncio
async def test_lookup_offer_async(transport: str='grpc_asyncio', request_type=service.LookupOfferRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.lookup_offer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(offers.Offer(name='name_value', deal_code='deal_code_value'))
        response = await client.lookup_offer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupOfferRequest()
    assert isinstance(response, offers.Offer)
    assert response.name == 'name_value'
    assert response.deal_code == 'deal_code_value'

@pytest.mark.asyncio
async def test_lookup_offer_async_from_dict():
    await test_lookup_offer_async(request_type=dict)

def test_lookup_offer_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.LookupOfferRequest()
    request.entitlement = 'entitlement_value'
    with mock.patch.object(type(client.transport.lookup_offer), '__call__') as call:
        call.return_value = offers.Offer()
        client.lookup_offer(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'entitlement=entitlement_value') in kw['metadata']

@pytest.mark.asyncio
async def test_lookup_offer_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.LookupOfferRequest()
    request.entitlement = 'entitlement_value'
    with mock.patch.object(type(client.transport.lookup_offer), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(offers.Offer())
        await client.lookup_offer(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'entitlement=entitlement_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ListProductsRequest, dict])
def test_list_products(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = service.ListProductsResponse(next_page_token='next_page_token_value')
        response = client.list_products(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListProductsRequest()
    assert isinstance(response, pagers.ListProductsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_products_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        client.list_products()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListProductsRequest()

@pytest.mark.asyncio
async def test_list_products_async(transport: str='grpc_asyncio', request_type=service.ListProductsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListProductsResponse(next_page_token='next_page_token_value'))
        response = await client.list_products(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListProductsRequest()
    assert isinstance(response, pagers.ListProductsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_products_async_from_dict():
    await test_list_products_async(request_type=dict)

def test_list_products_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.side_effect = (service.ListProductsResponse(products=[products.Product(), products.Product(), products.Product()], next_page_token='abc'), service.ListProductsResponse(products=[], next_page_token='def'), service.ListProductsResponse(products=[products.Product()], next_page_token='ghi'), service.ListProductsResponse(products=[products.Product(), products.Product()]), RuntimeError)
        metadata = ()
        pager = client.list_products(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, products.Product) for i in results))

def test_list_products_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_products), '__call__') as call:
        call.side_effect = (service.ListProductsResponse(products=[products.Product(), products.Product(), products.Product()], next_page_token='abc'), service.ListProductsResponse(products=[], next_page_token='def'), service.ListProductsResponse(products=[products.Product()], next_page_token='ghi'), service.ListProductsResponse(products=[products.Product(), products.Product()]), RuntimeError)
        pages = list(client.list_products(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_products_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_products), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListProductsResponse(products=[products.Product(), products.Product(), products.Product()], next_page_token='abc'), service.ListProductsResponse(products=[], next_page_token='def'), service.ListProductsResponse(products=[products.Product()], next_page_token='ghi'), service.ListProductsResponse(products=[products.Product(), products.Product()]), RuntimeError)
        async_pager = await client.list_products(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, products.Product) for i in responses))

@pytest.mark.asyncio
async def test_list_products_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_products), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListProductsResponse(products=[products.Product(), products.Product(), products.Product()], next_page_token='abc'), service.ListProductsResponse(products=[], next_page_token='def'), service.ListProductsResponse(products=[products.Product()], next_page_token='ghi'), service.ListProductsResponse(products=[products.Product(), products.Product()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_products(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListSkusRequest, dict])
def test_list_skus(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        call.return_value = service.ListSkusResponse(next_page_token='next_page_token_value')
        response = client.list_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkusRequest()
    assert isinstance(response, pagers.ListSkusPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_skus_empty_call():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        client.list_skus()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkusRequest()

@pytest.mark.asyncio
async def test_list_skus_async(transport: str='grpc_asyncio', request_type=service.ListSkusRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkusResponse(next_page_token='next_page_token_value'))
        response = await client.list_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSkusRequest()
    assert isinstance(response, pagers.ListSkusAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_skus_async_from_dict():
    await test_list_skus_async(request_type=dict)

def test_list_skus_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSkusRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        call.return_value = service.ListSkusResponse()
        client.list_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_skus_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSkusRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSkusResponse())
        await client.list_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_skus_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        call.side_effect = (service.ListSkusResponse(skus=[products.Sku(), products.Sku(), products.Sku()], next_page_token='abc'), service.ListSkusResponse(skus=[], next_page_token='def'), service.ListSkusResponse(skus=[products.Sku()], next_page_token='ghi'), service.ListSkusResponse(skus=[products.Sku(), products.Sku()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_skus(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, products.Sku) for i in results))

def test_list_skus_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_skus), '__call__') as call:
        call.side_effect = (service.ListSkusResponse(skus=[products.Sku(), products.Sku(), products.Sku()], next_page_token='abc'), service.ListSkusResponse(skus=[], next_page_token='def'), service.ListSkusResponse(skus=[products.Sku()], next_page_token='ghi'), service.ListSkusResponse(skus=[products.Sku(), products.Sku()]), RuntimeError)
        pages = list(client.list_skus(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_skus_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSkusResponse(skus=[products.Sku(), products.Sku(), products.Sku()], next_page_token='abc'), service.ListSkusResponse(skus=[], next_page_token='def'), service.ListSkusResponse(skus=[products.Sku()], next_page_token='ghi'), service.ListSkusResponse(skus=[products.Sku(), products.Sku()]), RuntimeError)
        async_pager = await client.list_skus(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, products.Sku) for i in responses))

@pytest.mark.asyncio
async def test_list_skus_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSkusResponse(skus=[products.Sku(), products.Sku(), products.Sku()], next_page_token='abc'), service.ListSkusResponse(skus=[], next_page_token='def'), service.ListSkusResponse(skus=[products.Sku()], next_page_token='ghi'), service.ListSkusResponse(skus=[products.Sku(), products.Sku()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_skus(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListOffersRequest, dict])
def test_list_offers(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        call.return_value = service.ListOffersResponse(next_page_token='next_page_token_value')
        response = client.list_offers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListOffersRequest()
    assert isinstance(response, pagers.ListOffersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_offers_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        client.list_offers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListOffersRequest()

@pytest.mark.asyncio
async def test_list_offers_async(transport: str='grpc_asyncio', request_type=service.ListOffersRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListOffersResponse(next_page_token='next_page_token_value'))
        response = await client.list_offers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListOffersRequest()
    assert isinstance(response, pagers.ListOffersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_offers_async_from_dict():
    await test_list_offers_async(request_type=dict)

def test_list_offers_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListOffersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        call.return_value = service.ListOffersResponse()
        client.list_offers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_offers_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListOffersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListOffersResponse())
        await client.list_offers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_offers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        call.side_effect = (service.ListOffersResponse(offers=[offers.Offer(), offers.Offer(), offers.Offer()], next_page_token='abc'), service.ListOffersResponse(offers=[], next_page_token='def'), service.ListOffersResponse(offers=[offers.Offer()], next_page_token='ghi'), service.ListOffersResponse(offers=[offers.Offer(), offers.Offer()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_offers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, offers.Offer) for i in results))

def test_list_offers_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_offers), '__call__') as call:
        call.side_effect = (service.ListOffersResponse(offers=[offers.Offer(), offers.Offer(), offers.Offer()], next_page_token='abc'), service.ListOffersResponse(offers=[], next_page_token='def'), service.ListOffersResponse(offers=[offers.Offer()], next_page_token='ghi'), service.ListOffersResponse(offers=[offers.Offer(), offers.Offer()]), RuntimeError)
        pages = list(client.list_offers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_offers_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_offers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListOffersResponse(offers=[offers.Offer(), offers.Offer(), offers.Offer()], next_page_token='abc'), service.ListOffersResponse(offers=[], next_page_token='def'), service.ListOffersResponse(offers=[offers.Offer()], next_page_token='ghi'), service.ListOffersResponse(offers=[offers.Offer(), offers.Offer()]), RuntimeError)
        async_pager = await client.list_offers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, offers.Offer) for i in responses))

@pytest.mark.asyncio
async def test_list_offers_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_offers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListOffersResponse(offers=[offers.Offer(), offers.Offer(), offers.Offer()], next_page_token='abc'), service.ListOffersResponse(offers=[], next_page_token='def'), service.ListOffersResponse(offers=[offers.Offer()], next_page_token='ghi'), service.ListOffersResponse(offers=[offers.Offer(), offers.Offer()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_offers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListPurchasableSkusRequest, dict])
def test_list_purchasable_skus(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        call.return_value = service.ListPurchasableSkusResponse(next_page_token='next_page_token_value')
        response = client.list_purchasable_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListPurchasableSkusRequest()
    assert isinstance(response, pagers.ListPurchasableSkusPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_purchasable_skus_empty_call():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        client.list_purchasable_skus()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListPurchasableSkusRequest()

@pytest.mark.asyncio
async def test_list_purchasable_skus_async(transport: str='grpc_asyncio', request_type=service.ListPurchasableSkusRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListPurchasableSkusResponse(next_page_token='next_page_token_value'))
        response = await client.list_purchasable_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListPurchasableSkusRequest()
    assert isinstance(response, pagers.ListPurchasableSkusAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_purchasable_skus_async_from_dict():
    await test_list_purchasable_skus_async(request_type=dict)

def test_list_purchasable_skus_field_headers():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListPurchasableSkusRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        call.return_value = service.ListPurchasableSkusResponse()
        client.list_purchasable_skus(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_purchasable_skus_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListPurchasableSkusRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListPurchasableSkusResponse())
        await client.list_purchasable_skus(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

def test_list_purchasable_skus_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        call.side_effect = (service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku(), service.PurchasableSku()], next_page_token='abc'), service.ListPurchasableSkusResponse(purchasable_skus=[], next_page_token='def'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku()], next_page_token='ghi'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('customer', ''),)),)
        pager = client.list_purchasable_skus(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.PurchasableSku) for i in results))

def test_list_purchasable_skus_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__') as call:
        call.side_effect = (service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku(), service.PurchasableSku()], next_page_token='abc'), service.ListPurchasableSkusResponse(purchasable_skus=[], next_page_token='def'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku()], next_page_token='ghi'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku()]), RuntimeError)
        pages = list(client.list_purchasable_skus(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_purchasable_skus_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku(), service.PurchasableSku()], next_page_token='abc'), service.ListPurchasableSkusResponse(purchasable_skus=[], next_page_token='def'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku()], next_page_token='ghi'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku()]), RuntimeError)
        async_pager = await client.list_purchasable_skus(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.PurchasableSku) for i in responses))

@pytest.mark.asyncio
async def test_list_purchasable_skus_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_purchasable_skus), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku(), service.PurchasableSku()], next_page_token='abc'), service.ListPurchasableSkusResponse(purchasable_skus=[], next_page_token='def'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku()], next_page_token='ghi'), service.ListPurchasableSkusResponse(purchasable_skus=[service.PurchasableSku(), service.PurchasableSku()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_purchasable_skus(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListPurchasableOffersRequest, dict])
def test_list_purchasable_offers(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        call.return_value = service.ListPurchasableOffersResponse(next_page_token='next_page_token_value')
        response = client.list_purchasable_offers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListPurchasableOffersRequest()
    assert isinstance(response, pagers.ListPurchasableOffersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_purchasable_offers_empty_call():
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        client.list_purchasable_offers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListPurchasableOffersRequest()

@pytest.mark.asyncio
async def test_list_purchasable_offers_async(transport: str='grpc_asyncio', request_type=service.ListPurchasableOffersRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListPurchasableOffersResponse(next_page_token='next_page_token_value'))
        response = await client.list_purchasable_offers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListPurchasableOffersRequest()
    assert isinstance(response, pagers.ListPurchasableOffersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_purchasable_offers_async_from_dict():
    await test_list_purchasable_offers_async(request_type=dict)

def test_list_purchasable_offers_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListPurchasableOffersRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        call.return_value = service.ListPurchasableOffersResponse()
        client.list_purchasable_offers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_purchasable_offers_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListPurchasableOffersRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListPurchasableOffersResponse())
        await client.list_purchasable_offers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

def test_list_purchasable_offers_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        call.side_effect = (service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer(), service.PurchasableOffer()], next_page_token='abc'), service.ListPurchasableOffersResponse(purchasable_offers=[], next_page_token='def'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer()], next_page_token='ghi'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('customer', ''),)),)
        pager = client.list_purchasable_offers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.PurchasableOffer) for i in results))

def test_list_purchasable_offers_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__') as call:
        call.side_effect = (service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer(), service.PurchasableOffer()], next_page_token='abc'), service.ListPurchasableOffersResponse(purchasable_offers=[], next_page_token='def'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer()], next_page_token='ghi'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer()]), RuntimeError)
        pages = list(client.list_purchasable_offers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_purchasable_offers_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer(), service.PurchasableOffer()], next_page_token='abc'), service.ListPurchasableOffersResponse(purchasable_offers=[], next_page_token='def'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer()], next_page_token='ghi'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer()]), RuntimeError)
        async_pager = await client.list_purchasable_offers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.PurchasableOffer) for i in responses))

@pytest.mark.asyncio
async def test_list_purchasable_offers_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_purchasable_offers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer(), service.PurchasableOffer()], next_page_token='abc'), service.ListPurchasableOffersResponse(purchasable_offers=[], next_page_token='def'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer()], next_page_token='ghi'), service.ListPurchasableOffersResponse(purchasable_offers=[service.PurchasableOffer(), service.PurchasableOffer()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_purchasable_offers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.QueryEligibleBillingAccountsRequest, dict])
def test_query_eligible_billing_accounts(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.query_eligible_billing_accounts), '__call__') as call:
        call.return_value = service.QueryEligibleBillingAccountsResponse()
        response = client.query_eligible_billing_accounts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.QueryEligibleBillingAccountsRequest()
    assert isinstance(response, service.QueryEligibleBillingAccountsResponse)

def test_query_eligible_billing_accounts_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.query_eligible_billing_accounts), '__call__') as call:
        client.query_eligible_billing_accounts()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.QueryEligibleBillingAccountsRequest()

@pytest.mark.asyncio
async def test_query_eligible_billing_accounts_async(transport: str='grpc_asyncio', request_type=service.QueryEligibleBillingAccountsRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.query_eligible_billing_accounts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.QueryEligibleBillingAccountsResponse())
        response = await client.query_eligible_billing_accounts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.QueryEligibleBillingAccountsRequest()
    assert isinstance(response, service.QueryEligibleBillingAccountsResponse)

@pytest.mark.asyncio
async def test_query_eligible_billing_accounts_async_from_dict():
    await test_query_eligible_billing_accounts_async(request_type=dict)

def test_query_eligible_billing_accounts_field_headers():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.QueryEligibleBillingAccountsRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.query_eligible_billing_accounts), '__call__') as call:
        call.return_value = service.QueryEligibleBillingAccountsResponse()
        client.query_eligible_billing_accounts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

@pytest.mark.asyncio
async def test_query_eligible_billing_accounts_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.QueryEligibleBillingAccountsRequest()
    request.customer = 'customer_value'
    with mock.patch.object(type(client.transport.query_eligible_billing_accounts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.QueryEligibleBillingAccountsResponse())
        await client.query_eligible_billing_accounts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'customer=customer_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.RegisterSubscriberRequest, dict])
def test_register_subscriber(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.register_subscriber), '__call__') as call:
        call.return_value = service.RegisterSubscriberResponse(topic='topic_value')
        response = client.register_subscriber(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RegisterSubscriberRequest()
    assert isinstance(response, service.RegisterSubscriberResponse)
    assert response.topic == 'topic_value'

def test_register_subscriber_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.register_subscriber), '__call__') as call:
        client.register_subscriber()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RegisterSubscriberRequest()

@pytest.mark.asyncio
async def test_register_subscriber_async(transport: str='grpc_asyncio', request_type=service.RegisterSubscriberRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.register_subscriber), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.RegisterSubscriberResponse(topic='topic_value'))
        response = await client.register_subscriber(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RegisterSubscriberRequest()
    assert isinstance(response, service.RegisterSubscriberResponse)
    assert response.topic == 'topic_value'

@pytest.mark.asyncio
async def test_register_subscriber_async_from_dict():
    await test_register_subscriber_async(request_type=dict)

def test_register_subscriber_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RegisterSubscriberRequest()
    request.account = 'account_value'
    with mock.patch.object(type(client.transport.register_subscriber), '__call__') as call:
        call.return_value = service.RegisterSubscriberResponse()
        client.register_subscriber(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'account=account_value') in kw['metadata']

@pytest.mark.asyncio
async def test_register_subscriber_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RegisterSubscriberRequest()
    request.account = 'account_value'
    with mock.patch.object(type(client.transport.register_subscriber), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.RegisterSubscriberResponse())
        await client.register_subscriber(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'account=account_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.UnregisterSubscriberRequest, dict])
def test_unregister_subscriber(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.unregister_subscriber), '__call__') as call:
        call.return_value = service.UnregisterSubscriberResponse(topic='topic_value')
        response = client.unregister_subscriber(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UnregisterSubscriberRequest()
    assert isinstance(response, service.UnregisterSubscriberResponse)
    assert response.topic == 'topic_value'

def test_unregister_subscriber_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.unregister_subscriber), '__call__') as call:
        client.unregister_subscriber()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UnregisterSubscriberRequest()

@pytest.mark.asyncio
async def test_unregister_subscriber_async(transport: str='grpc_asyncio', request_type=service.UnregisterSubscriberRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.unregister_subscriber), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.UnregisterSubscriberResponse(topic='topic_value'))
        response = await client.unregister_subscriber(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UnregisterSubscriberRequest()
    assert isinstance(response, service.UnregisterSubscriberResponse)
    assert response.topic == 'topic_value'

@pytest.mark.asyncio
async def test_unregister_subscriber_async_from_dict():
    await test_unregister_subscriber_async(request_type=dict)

def test_unregister_subscriber_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UnregisterSubscriberRequest()
    request.account = 'account_value'
    with mock.patch.object(type(client.transport.unregister_subscriber), '__call__') as call:
        call.return_value = service.UnregisterSubscriberResponse()
        client.unregister_subscriber(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'account=account_value') in kw['metadata']

@pytest.mark.asyncio
async def test_unregister_subscriber_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UnregisterSubscriberRequest()
    request.account = 'account_value'
    with mock.patch.object(type(client.transport.unregister_subscriber), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.UnregisterSubscriberResponse())
        await client.unregister_subscriber(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'account=account_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ListSubscribersRequest, dict])
def test_list_subscribers(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        call.return_value = service.ListSubscribersResponse(topic='topic_value', service_accounts=['service_accounts_value'], next_page_token='next_page_token_value')
        response = client.list_subscribers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSubscribersRequest()
    assert isinstance(response, pagers.ListSubscribersPager)
    assert response.topic == 'topic_value'
    assert response.service_accounts == ['service_accounts_value']
    assert response.next_page_token == 'next_page_token_value'

def test_list_subscribers_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        client.list_subscribers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSubscribersRequest()

@pytest.mark.asyncio
async def test_list_subscribers_async(transport: str='grpc_asyncio', request_type=service.ListSubscribersRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSubscribersResponse(topic='topic_value', service_accounts=['service_accounts_value'], next_page_token='next_page_token_value'))
        response = await client.list_subscribers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSubscribersRequest()
    assert isinstance(response, pagers.ListSubscribersAsyncPager)
    assert response.topic == 'topic_value'
    assert response.service_accounts == ['service_accounts_value']
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_subscribers_async_from_dict():
    await test_list_subscribers_async(request_type=dict)

def test_list_subscribers_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSubscribersRequest()
    request.account = 'account_value'
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        call.return_value = service.ListSubscribersResponse()
        client.list_subscribers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'account=account_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_subscribers_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSubscribersRequest()
    request.account = 'account_value'
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSubscribersResponse())
        await client.list_subscribers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'account=account_value') in kw['metadata']

def test_list_subscribers_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        call.side_effect = (service.ListSubscribersResponse(service_accounts=[str(), str(), str()], next_page_token='abc'), service.ListSubscribersResponse(service_accounts=[], next_page_token='def'), service.ListSubscribersResponse(service_accounts=[str()], next_page_token='ghi'), service.ListSubscribersResponse(service_accounts=[str(), str()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('account', ''),)),)
        pager = client.list_subscribers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, str) for i in results))

def test_list_subscribers_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_subscribers), '__call__') as call:
        call.side_effect = (service.ListSubscribersResponse(service_accounts=[str(), str(), str()], next_page_token='abc'), service.ListSubscribersResponse(service_accounts=[], next_page_token='def'), service.ListSubscribersResponse(service_accounts=[str()], next_page_token='ghi'), service.ListSubscribersResponse(service_accounts=[str(), str()]), RuntimeError)
        pages = list(client.list_subscribers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_subscribers_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_subscribers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSubscribersResponse(service_accounts=[str(), str(), str()], next_page_token='abc'), service.ListSubscribersResponse(service_accounts=[], next_page_token='def'), service.ListSubscribersResponse(service_accounts=[str()], next_page_token='ghi'), service.ListSubscribersResponse(service_accounts=[str(), str()]), RuntimeError)
        async_pager = await client.list_subscribers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, str) for i in responses))

@pytest.mark.asyncio
async def test_list_subscribers_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_subscribers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSubscribersResponse(service_accounts=[str(), str(), str()], next_page_token='abc'), service.ListSubscribersResponse(service_accounts=[], next_page_token='def'), service.ListSubscribersResponse(service_accounts=[str()], next_page_token='ghi'), service.ListSubscribersResponse(service_accounts=[str(), str()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_subscribers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListEntitlementChangesRequest, dict])
def test_list_entitlement_changes(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.return_value = service.ListEntitlementChangesResponse(next_page_token='next_page_token_value')
        response = client.list_entitlement_changes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntitlementChangesRequest()
    assert isinstance(response, pagers.ListEntitlementChangesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entitlement_changes_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        client.list_entitlement_changes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntitlementChangesRequest()

@pytest.mark.asyncio
async def test_list_entitlement_changes_async(transport: str='grpc_asyncio', request_type=service.ListEntitlementChangesRequest):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntitlementChangesResponse(next_page_token='next_page_token_value'))
        response = await client.list_entitlement_changes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntitlementChangesRequest()
    assert isinstance(response, pagers.ListEntitlementChangesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_entitlement_changes_async_from_dict():
    await test_list_entitlement_changes_async(request_type=dict)

def test_list_entitlement_changes_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEntitlementChangesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.return_value = service.ListEntitlementChangesResponse()
        client.list_entitlement_changes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_entitlement_changes_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEntitlementChangesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntitlementChangesResponse())
        await client.list_entitlement_changes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_entitlement_changes_flattened():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.return_value = service.ListEntitlementChangesResponse()
        client.list_entitlement_changes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_entitlement_changes_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_entitlement_changes(service.ListEntitlementChangesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_entitlement_changes_flattened_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.return_value = service.ListEntitlementChangesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntitlementChangesResponse())
        response = await client.list_entitlement_changes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_entitlement_changes_flattened_error_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_entitlement_changes(service.ListEntitlementChangesRequest(), parent='parent_value')

def test_list_entitlement_changes_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.side_effect = (service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()], next_page_token='abc'), service.ListEntitlementChangesResponse(entitlement_changes=[], next_page_token='def'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange()], next_page_token='ghi'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_entitlement_changes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, entitlement_changes.EntitlementChange) for i in results))

def test_list_entitlement_changes_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__') as call:
        call.side_effect = (service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()], next_page_token='abc'), service.ListEntitlementChangesResponse(entitlement_changes=[], next_page_token='def'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange()], next_page_token='ghi'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()]), RuntimeError)
        pages = list(client.list_entitlement_changes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_entitlement_changes_async_pager():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()], next_page_token='abc'), service.ListEntitlementChangesResponse(entitlement_changes=[], next_page_token='def'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange()], next_page_token='ghi'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()]), RuntimeError)
        async_pager = await client.list_entitlement_changes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, entitlement_changes.EntitlementChange) for i in responses))

@pytest.mark.asyncio
async def test_list_entitlement_changes_async_pages():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entitlement_changes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()], next_page_token='abc'), service.ListEntitlementChangesResponse(entitlement_changes=[], next_page_token='def'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange()], next_page_token='ghi'), service.ListEntitlementChangesResponse(entitlement_changes=[entitlement_changes.EntitlementChange(), entitlement_changes.EntitlementChange()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_entitlement_changes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.CloudChannelServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudChannelServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudChannelServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudChannelServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudChannelServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudChannelServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudChannelServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudChannelServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudChannelServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudChannelServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.CloudChannelServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudChannelServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudChannelServiceGrpcTransport, transports.CloudChannelServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = CloudChannelServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudChannelServiceGrpcTransport)

def test_cloud_channel_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudChannelServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_channel_service_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.channel_v1.services.cloud_channel_service.transports.CloudChannelServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudChannelServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_customers', 'get_customer', 'check_cloud_identity_accounts_exist', 'create_customer', 'update_customer', 'delete_customer', 'import_customer', 'provision_cloud_identity', 'list_entitlements', 'list_transferable_skus', 'list_transferable_offers', 'get_entitlement', 'create_entitlement', 'change_parameters', 'change_renewal_settings', 'change_offer', 'start_paid_service', 'suspend_entitlement', 'cancel_entitlement', 'activate_entitlement', 'transfer_entitlements', 'transfer_entitlements_to_google', 'list_channel_partner_links', 'get_channel_partner_link', 'create_channel_partner_link', 'update_channel_partner_link', 'get_customer_repricing_config', 'list_customer_repricing_configs', 'create_customer_repricing_config', 'update_customer_repricing_config', 'delete_customer_repricing_config', 'get_channel_partner_repricing_config', 'list_channel_partner_repricing_configs', 'create_channel_partner_repricing_config', 'update_channel_partner_repricing_config', 'delete_channel_partner_repricing_config', 'list_sku_groups', 'list_sku_group_billable_skus', 'lookup_offer', 'list_products', 'list_skus', 'list_offers', 'list_purchasable_skus', 'list_purchasable_offers', 'query_eligible_billing_accounts', 'register_subscriber', 'unregister_subscriber', 'list_subscribers', 'list_entitlement_changes', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_cloud_channel_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.channel_v1.services.cloud_channel_service.transports.CloudChannelServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudChannelServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/apps.order',), quota_project_id='octopus')

def test_cloud_channel_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.channel_v1.services.cloud_channel_service.transports.CloudChannelServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudChannelServiceTransport()
        adc.assert_called_once()

def test_cloud_channel_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudChannelServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/apps.order',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudChannelServiceGrpcTransport, transports.CloudChannelServiceGrpcAsyncIOTransport])
def test_cloud_channel_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/apps.order',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudChannelServiceGrpcTransport, transports.CloudChannelServiceGrpcAsyncIOTransport])
def test_cloud_channel_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudChannelServiceGrpcTransport, grpc_helpers), (transports.CloudChannelServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_channel_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudchannel.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/apps.order',), scopes=['1', '2'], default_host='cloudchannel.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudChannelServiceGrpcTransport, transports.CloudChannelServiceGrpcAsyncIOTransport])
def test_cloud_channel_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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
def test_cloud_channel_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudchannel.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'cloudchannel.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_cloud_channel_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudchannel.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'cloudchannel.googleapis.com:8000'

def test_cloud_channel_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudChannelServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_channel_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudChannelServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudChannelServiceGrpcTransport, transports.CloudChannelServiceGrpcAsyncIOTransport])
def test_cloud_channel_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CloudChannelServiceGrpcTransport, transports.CloudChannelServiceGrpcAsyncIOTransport])
def test_cloud_channel_service_transport_channel_mtls_with_adc(transport_class):
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

def test_cloud_channel_service_grpc_lro_client():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_channel_service_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_billing_account_path():
    if False:
        i = 10
        return i + 15
    account = 'squid'
    billing_account = 'clam'
    expected = 'accounts/{account}/billingAccounts/{billing_account}'.format(account=account, billing_account=billing_account)
    actual = CloudChannelServiceClient.billing_account_path(account, billing_account)
    assert expected == actual

def test_parse_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'account': 'whelk', 'billing_account': 'octopus'}
    path = CloudChannelServiceClient.billing_account_path(**expected)
    actual = CloudChannelServiceClient.parse_billing_account_path(path)
    assert expected == actual

def test_channel_partner_link_path():
    if False:
        while True:
            i = 10
    account = 'oyster'
    channel_partner_link = 'nudibranch'
    expected = 'accounts/{account}/channelPartnerLinks/{channel_partner_link}'.format(account=account, channel_partner_link=channel_partner_link)
    actual = CloudChannelServiceClient.channel_partner_link_path(account, channel_partner_link)
    assert expected == actual

def test_parse_channel_partner_link_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'account': 'cuttlefish', 'channel_partner_link': 'mussel'}
    path = CloudChannelServiceClient.channel_partner_link_path(**expected)
    actual = CloudChannelServiceClient.parse_channel_partner_link_path(path)
    assert expected == actual

def test_channel_partner_repricing_config_path():
    if False:
        i = 10
        return i + 15
    account = 'winkle'
    channel_partner = 'nautilus'
    channel_partner_repricing_config = 'scallop'
    expected = 'accounts/{account}/channelPartnerLinks/{channel_partner}/channelPartnerRepricingConfigs/{channel_partner_repricing_config}'.format(account=account, channel_partner=channel_partner, channel_partner_repricing_config=channel_partner_repricing_config)
    actual = CloudChannelServiceClient.channel_partner_repricing_config_path(account, channel_partner, channel_partner_repricing_config)
    assert expected == actual

def test_parse_channel_partner_repricing_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'account': 'abalone', 'channel_partner': 'squid', 'channel_partner_repricing_config': 'clam'}
    path = CloudChannelServiceClient.channel_partner_repricing_config_path(**expected)
    actual = CloudChannelServiceClient.parse_channel_partner_repricing_config_path(path)
    assert expected == actual

def test_customer_path():
    if False:
        return 10
    account = 'whelk'
    customer = 'octopus'
    expected = 'accounts/{account}/customers/{customer}'.format(account=account, customer=customer)
    actual = CloudChannelServiceClient.customer_path(account, customer)
    assert expected == actual

def test_parse_customer_path():
    if False:
        while True:
            i = 10
    expected = {'account': 'oyster', 'customer': 'nudibranch'}
    path = CloudChannelServiceClient.customer_path(**expected)
    actual = CloudChannelServiceClient.parse_customer_path(path)
    assert expected == actual

def test_customer_repricing_config_path():
    if False:
        while True:
            i = 10
    account = 'cuttlefish'
    customer = 'mussel'
    customer_repricing_config = 'winkle'
    expected = 'accounts/{account}/customers/{customer}/customerRepricingConfigs/{customer_repricing_config}'.format(account=account, customer=customer, customer_repricing_config=customer_repricing_config)
    actual = CloudChannelServiceClient.customer_repricing_config_path(account, customer, customer_repricing_config)
    assert expected == actual

def test_parse_customer_repricing_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'account': 'nautilus', 'customer': 'scallop', 'customer_repricing_config': 'abalone'}
    path = CloudChannelServiceClient.customer_repricing_config_path(**expected)
    actual = CloudChannelServiceClient.parse_customer_repricing_config_path(path)
    assert expected == actual

def test_entitlement_path():
    if False:
        return 10
    account = 'squid'
    customer = 'clam'
    entitlement = 'whelk'
    expected = 'accounts/{account}/customers/{customer}/entitlements/{entitlement}'.format(account=account, customer=customer, entitlement=entitlement)
    actual = CloudChannelServiceClient.entitlement_path(account, customer, entitlement)
    assert expected == actual

def test_parse_entitlement_path():
    if False:
        while True:
            i = 10
    expected = {'account': 'octopus', 'customer': 'oyster', 'entitlement': 'nudibranch'}
    path = CloudChannelServiceClient.entitlement_path(**expected)
    actual = CloudChannelServiceClient.parse_entitlement_path(path)
    assert expected == actual

def test_offer_path():
    if False:
        return 10
    account = 'cuttlefish'
    offer = 'mussel'
    expected = 'accounts/{account}/offers/{offer}'.format(account=account, offer=offer)
    actual = CloudChannelServiceClient.offer_path(account, offer)
    assert expected == actual

def test_parse_offer_path():
    if False:
        return 10
    expected = {'account': 'winkle', 'offer': 'nautilus'}
    path = CloudChannelServiceClient.offer_path(**expected)
    actual = CloudChannelServiceClient.parse_offer_path(path)
    assert expected == actual

def test_product_path():
    if False:
        i = 10
        return i + 15
    product = 'scallop'
    expected = 'products/{product}'.format(product=product)
    actual = CloudChannelServiceClient.product_path(product)
    assert expected == actual

def test_parse_product_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'product': 'abalone'}
    path = CloudChannelServiceClient.product_path(**expected)
    actual = CloudChannelServiceClient.parse_product_path(path)
    assert expected == actual

def test_sku_path():
    if False:
        while True:
            i = 10
    product = 'squid'
    sku = 'clam'
    expected = 'products/{product}/skus/{sku}'.format(product=product, sku=sku)
    actual = CloudChannelServiceClient.sku_path(product, sku)
    assert expected == actual

def test_parse_sku_path():
    if False:
        return 10
    expected = {'product': 'whelk', 'sku': 'octopus'}
    path = CloudChannelServiceClient.sku_path(**expected)
    actual = CloudChannelServiceClient.parse_sku_path(path)
    assert expected == actual

def test_sku_group_path():
    if False:
        return 10
    account = 'oyster'
    sku_group = 'nudibranch'
    expected = 'accounts/{account}/skuGroups/{sku_group}'.format(account=account, sku_group=sku_group)
    actual = CloudChannelServiceClient.sku_group_path(account, sku_group)
    assert expected == actual

def test_parse_sku_group_path():
    if False:
        return 10
    expected = {'account': 'cuttlefish', 'sku_group': 'mussel'}
    path = CloudChannelServiceClient.sku_group_path(**expected)
    actual = CloudChannelServiceClient.parse_sku_group_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudChannelServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'nautilus'}
    path = CloudChannelServiceClient.common_billing_account_path(**expected)
    actual = CloudChannelServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudChannelServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'abalone'}
    path = CloudChannelServiceClient.common_folder_path(**expected)
    actual = CloudChannelServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudChannelServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'clam'}
    path = CloudChannelServiceClient.common_organization_path(**expected)
    actual = CloudChannelServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudChannelServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus'}
    path = CloudChannelServiceClient.common_project_path(**expected)
    actual = CloudChannelServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudChannelServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = CloudChannelServiceClient.common_location_path(**expected)
    actual = CloudChannelServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudChannelServiceTransport, '_prep_wrapped_messages') as prep:
        client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudChannelServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudChannelServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

@pytest.mark.asyncio
async def test_delete_operation_async(transport: str='grpc'):
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

def test_delete_operation_field_headers():
    if False:
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_operation_field_headers_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_delete_operation_from_dict():
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CloudChannelServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = CloudChannelServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudChannelServiceClient, transports.CloudChannelServiceGrpcTransport), (CloudChannelServiceAsyncClient, transports.CloudChannelServiceGrpcAsyncIOTransport)])
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