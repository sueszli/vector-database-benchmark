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
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.retail_v2.services.search_service import SearchServiceAsyncClient, SearchServiceClient, pagers, transports
from google.cloud.retail_v2.types import common, search_service

def client_cert_source_callback():
    if False:
        print('Hello World!')
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
    assert SearchServiceClient._get_default_mtls_endpoint(None) is None
    assert SearchServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SearchServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SearchServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SearchServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SearchServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SearchServiceClient, 'grpc'), (SearchServiceAsyncClient, 'grpc_asyncio'), (SearchServiceClient, 'rest')])
def test_search_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SearchServiceGrpcTransport, 'grpc'), (transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.SearchServiceRestTransport, 'rest')])
def test_search_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(SearchServiceClient, 'grpc'), (SearchServiceAsyncClient, 'grpc_asyncio'), (SearchServiceClient, 'rest')])
def test_search_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

def test_search_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = SearchServiceClient.get_transport_class()
    available_transports = [transports.SearchServiceGrpcTransport, transports.SearchServiceRestTransport]
    assert transport in available_transports
    transport = SearchServiceClient.get_transport_class('grpc')
    assert transport == transports.SearchServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SearchServiceClient, transports.SearchServiceGrpcTransport, 'grpc'), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (SearchServiceClient, transports.SearchServiceRestTransport, 'rest')])
@mock.patch.object(SearchServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SearchServiceClient))
@mock.patch.object(SearchServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SearchServiceAsyncClient))
def test_search_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(SearchServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SearchServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SearchServiceClient, transports.SearchServiceGrpcTransport, 'grpc', 'true'), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (SearchServiceClient, transports.SearchServiceGrpcTransport, 'grpc', 'false'), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (SearchServiceClient, transports.SearchServiceRestTransport, 'rest', 'true'), (SearchServiceClient, transports.SearchServiceRestTransport, 'rest', 'false')])
@mock.patch.object(SearchServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SearchServiceClient))
@mock.patch.object(SearchServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SearchServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_search_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [SearchServiceClient, SearchServiceAsyncClient])
@mock.patch.object(SearchServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SearchServiceClient))
@mock.patch.object(SearchServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SearchServiceAsyncClient))
def test_search_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SearchServiceClient, transports.SearchServiceGrpcTransport, 'grpc'), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (SearchServiceClient, transports.SearchServiceRestTransport, 'rest')])
def test_search_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SearchServiceClient, transports.SearchServiceGrpcTransport, 'grpc', grpc_helpers), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (SearchServiceClient, transports.SearchServiceRestTransport, 'rest', None)])
def test_search_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_search_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.retail_v2.services.search_service.transports.SearchServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = SearchServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SearchServiceClient, transports.SearchServiceGrpcTransport, 'grpc', grpc_helpers), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_search_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [search_service.SearchRequest, dict])
def test_search(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = search_service.SearchResponse(total_size=1086, corrected_query='corrected_query_value', attribution_token='attribution_token_value', next_page_token='next_page_token_value', redirect_uri='redirect_uri_value', applied_controls=['applied_controls_value'])
        response = client.search(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == search_service.SearchRequest()
    assert isinstance(response, pagers.SearchPager)
    assert response.total_size == 1086
    assert response.corrected_query == 'corrected_query_value'
    assert response.attribution_token == 'attribution_token_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.redirect_uri == 'redirect_uri_value'
    assert response.applied_controls == ['applied_controls_value']

def test_search_empty_call():
    if False:
        while True:
            i = 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        client.search()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == search_service.SearchRequest()

@pytest.mark.asyncio
async def test_search_async(transport: str='grpc_asyncio', request_type=search_service.SearchRequest):
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(search_service.SearchResponse(total_size=1086, corrected_query='corrected_query_value', attribution_token='attribution_token_value', next_page_token='next_page_token_value', redirect_uri='redirect_uri_value', applied_controls=['applied_controls_value']))
        response = await client.search(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == search_service.SearchRequest()
    assert isinstance(response, pagers.SearchAsyncPager)
    assert response.total_size == 1086
    assert response.corrected_query == 'corrected_query_value'
    assert response.attribution_token == 'attribution_token_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.redirect_uri == 'redirect_uri_value'
    assert response.applied_controls == ['applied_controls_value']

@pytest.mark.asyncio
async def test_search_async_from_dict():
    await test_search_async(request_type=dict)

def test_search_field_headers():
    if False:
        while True:
            i = 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = search_service.SearchRequest()
    request.placement = 'placement_value'
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = search_service.SearchResponse()
        client.search(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'placement=placement_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_field_headers_async():
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = search_service.SearchRequest()
    request.placement = 'placement_value'
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(search_service.SearchResponse())
        await client.search(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'placement=placement_value') in kw['metadata']

def test_search_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.side_effect = (search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()], next_page_token='abc'), search_service.SearchResponse(results=[], next_page_token='def'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult()], next_page_token='ghi'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('placement', ''),)),)
        pager = client.search(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, search_service.SearchResponse.SearchResult) for i in results))

def test_search_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.side_effect = (search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()], next_page_token='abc'), search_service.SearchResponse(results=[], next_page_token='def'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult()], next_page_token='ghi'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()]), RuntimeError)
        pages = list(client.search(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_async_pager():
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()], next_page_token='abc'), search_service.SearchResponse(results=[], next_page_token='def'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult()], next_page_token='ghi'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()]), RuntimeError)
        async_pager = await client.search(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, search_service.SearchResponse.SearchResult) for i in responses))

@pytest.mark.asyncio
async def test_search_async_pages():
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()], next_page_token='abc'), search_service.SearchResponse(results=[], next_page_token='def'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult()], next_page_token='ghi'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()]), RuntimeError)
        pages = []
        async for page_ in (await client.search(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [search_service.SearchRequest, dict])
def test_search_rest(request_type):
    if False:
        print('Hello World!')
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'placement': 'projects/sample1/locations/sample2/catalogs/sample3/placements/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = search_service.SearchResponse(total_size=1086, corrected_query='corrected_query_value', attribution_token='attribution_token_value', next_page_token='next_page_token_value', redirect_uri='redirect_uri_value', applied_controls=['applied_controls_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = search_service.SearchResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search(request)
    assert isinstance(response, pagers.SearchPager)
    assert response.total_size == 1086
    assert response.corrected_query == 'corrected_query_value'
    assert response.attribution_token == 'attribution_token_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.redirect_uri == 'redirect_uri_value'
    assert response.applied_controls == ['applied_controls_value']

def test_search_rest_required_fields(request_type=search_service.SearchRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SearchServiceRestTransport
    request_init = {}
    request_init['placement'] = ''
    request_init['visitor_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['placement'] = 'placement_value'
    jsonified_request['visitorId'] = 'visitor_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'placement' in jsonified_request
    assert jsonified_request['placement'] == 'placement_value'
    assert 'visitorId' in jsonified_request
    assert jsonified_request['visitorId'] == 'visitor_id_value'
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = search_service.SearchResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = search_service.SearchResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SearchServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('placement', 'visitorId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SearchServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SearchServiceRestInterceptor())
    client = SearchServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SearchServiceRestInterceptor, 'post_search') as post, mock.patch.object(transports.SearchServiceRestInterceptor, 'pre_search') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = search_service.SearchRequest.pb(search_service.SearchRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = search_service.SearchResponse.to_json(search_service.SearchResponse())
        request = search_service.SearchRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = search_service.SearchResponse()
        client.search(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_rest_bad_request(transport: str='rest', request_type=search_service.SearchRequest):
    if False:
        i = 10
        return i + 15
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'placement': 'projects/sample1/locations/sample2/catalogs/sample3/placements/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search(request)

def test_search_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()], next_page_token='abc'), search_service.SearchResponse(results=[], next_page_token='def'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult()], next_page_token='ghi'), search_service.SearchResponse(results=[search_service.SearchResponse.SearchResult(), search_service.SearchResponse.SearchResult()]))
        response = response + response
        response = tuple((search_service.SearchResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'placement': 'projects/sample1/locations/sample2/catalogs/sample3/placements/sample4'}
        pager = client.search(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, search_service.SearchResponse.SearchResult) for i in results))
        pages = list(client.search(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.SearchServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SearchServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SearchServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SearchServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SearchServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SearchServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SearchServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SearchServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.SearchServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SearchServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SearchServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.SearchServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.SearchServiceGrpcTransport, transports.SearchServiceGrpcAsyncIOTransport, transports.SearchServiceRestTransport])
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
        i = 10
        return i + 15
    transport = SearchServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.SearchServiceGrpcTransport)

def test_search_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SearchServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_search_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.retail_v2.services.search_service.transports.SearchServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SearchServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('search', 'get_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_search_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.retail_v2.services.search_service.transports.SearchServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SearchServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_search_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.retail_v2.services.search_service.transports.SearchServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SearchServiceTransport()
        adc.assert_called_once()

def test_search_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SearchServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.SearchServiceGrpcTransport, transports.SearchServiceGrpcAsyncIOTransport])
def test_search_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.SearchServiceGrpcTransport, transports.SearchServiceGrpcAsyncIOTransport, transports.SearchServiceRestTransport])
def test_search_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.SearchServiceGrpcTransport, grpc_helpers), (transports.SearchServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_search_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.SearchServiceGrpcTransport, transports.SearchServiceGrpcAsyncIOTransport])
def test_search_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_search_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.SearchServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_search_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_search_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_search_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = SearchServiceClient(credentials=creds1, transport=transport_name)
    client2 = SearchServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.search._session
    session2 = client2.transport.search._session
    assert session1 != session2

def test_search_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SearchServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_search_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SearchServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.SearchServiceGrpcTransport, transports.SearchServiceGrpcAsyncIOTransport])
def test_search_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.SearchServiceGrpcTransport, transports.SearchServiceGrpcAsyncIOTransport])
def test_search_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        return 10
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

def test_branch_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    catalog = 'whelk'
    branch = 'octopus'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/branches/{branch}'.format(project=project, location=location, catalog=catalog, branch=branch)
    actual = SearchServiceClient.branch_path(project, location, catalog, branch)
    assert expected == actual

def test_parse_branch_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'catalog': 'cuttlefish', 'branch': 'mussel'}
    path = SearchServiceClient.branch_path(**expected)
    actual = SearchServiceClient.parse_branch_path(path)
    assert expected == actual

def test_experiment_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    catalog = 'scallop'
    experiment = 'abalone'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/experiments/{experiment}'.format(project=project, location=location, catalog=catalog, experiment=experiment)
    actual = SearchServiceClient.experiment_path(project, location, catalog, experiment)
    assert expected == actual

def test_parse_experiment_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam', 'catalog': 'whelk', 'experiment': 'octopus'}
    path = SearchServiceClient.experiment_path(**expected)
    actual = SearchServiceClient.parse_experiment_path(path)
    assert expected == actual

def test_product_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    catalog = 'cuttlefish'
    branch = 'mussel'
    product = 'winkle'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/branches/{branch}/products/{product}'.format(project=project, location=location, catalog=catalog, branch=branch, product=product)
    actual = SearchServiceClient.product_path(project, location, catalog, branch, product)
    assert expected == actual

def test_parse_product_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'catalog': 'abalone', 'branch': 'squid', 'product': 'clam'}
    path = SearchServiceClient.product_path(**expected)
    actual = SearchServiceClient.parse_product_path(path)
    assert expected == actual

def test_serving_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    catalog = 'oyster'
    serving_config = 'nudibranch'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/servingConfigs/{serving_config}'.format(project=project, location=location, catalog=catalog, serving_config=serving_config)
    actual = SearchServiceClient.serving_config_path(project, location, catalog, serving_config)
    assert expected == actual

def test_parse_serving_config_path():
    if False:
        return 10
    expected = {'project': 'cuttlefish', 'location': 'mussel', 'catalog': 'winkle', 'serving_config': 'nautilus'}
    path = SearchServiceClient.serving_config_path(**expected)
    actual = SearchServiceClient.parse_serving_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SearchServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'abalone'}
    path = SearchServiceClient.common_billing_account_path(**expected)
    actual = SearchServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SearchServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = SearchServiceClient.common_folder_path(**expected)
    actual = SearchServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SearchServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'octopus'}
    path = SearchServiceClient.common_organization_path(**expected)
    actual = SearchServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = SearchServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'nudibranch'}
    path = SearchServiceClient.common_project_path(**expected)
    actual = SearchServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SearchServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = SearchServiceClient.common_location_path(**expected)
    actual = SearchServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SearchServiceTransport, '_prep_wrapped_messages') as prep:
        client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SearchServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = SearchServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = SearchServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = SearchServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SearchServiceClient, transports.SearchServiceGrpcTransport), (SearchServiceAsyncClient, transports.SearchServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)