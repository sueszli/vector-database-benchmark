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
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.monitoring_dashboard_v1.services.dashboards_service import DashboardsServiceAsyncClient, DashboardsServiceClient, pagers, transports
from google.cloud.monitoring_dashboard_v1.types import alertchart, collapsible_group, common, dashboard, dashboard_filter, dashboards_service, layouts, logs_panel, metrics, scorecard, table, table_display_options, text, widget, xychart

def client_cert_source_callback():
    if False:
        print('Hello World!')
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
    assert DashboardsServiceClient._get_default_mtls_endpoint(None) is None
    assert DashboardsServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DashboardsServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DashboardsServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DashboardsServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DashboardsServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DashboardsServiceClient, 'grpc'), (DashboardsServiceAsyncClient, 'grpc_asyncio'), (DashboardsServiceClient, 'rest')])
def test_dashboards_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('monitoring.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://monitoring.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DashboardsServiceGrpcTransport, 'grpc'), (transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.DashboardsServiceRestTransport, 'rest')])
def test_dashboards_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DashboardsServiceClient, 'grpc'), (DashboardsServiceAsyncClient, 'grpc_asyncio'), (DashboardsServiceClient, 'rest')])
def test_dashboards_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('monitoring.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://monitoring.googleapis.com')

def test_dashboards_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = DashboardsServiceClient.get_transport_class()
    available_transports = [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceRestTransport]
    assert transport in available_transports
    transport = DashboardsServiceClient.get_transport_class('grpc')
    assert transport == transports.DashboardsServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DashboardsServiceClient, transports.DashboardsServiceGrpcTransport, 'grpc'), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DashboardsServiceClient, transports.DashboardsServiceRestTransport, 'rest')])
@mock.patch.object(DashboardsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DashboardsServiceClient))
@mock.patch.object(DashboardsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DashboardsServiceAsyncClient))
def test_dashboards_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(DashboardsServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DashboardsServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DashboardsServiceClient, transports.DashboardsServiceGrpcTransport, 'grpc', 'true'), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DashboardsServiceClient, transports.DashboardsServiceGrpcTransport, 'grpc', 'false'), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (DashboardsServiceClient, transports.DashboardsServiceRestTransport, 'rest', 'true'), (DashboardsServiceClient, transports.DashboardsServiceRestTransport, 'rest', 'false')])
@mock.patch.object(DashboardsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DashboardsServiceClient))
@mock.patch.object(DashboardsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DashboardsServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_dashboards_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [DashboardsServiceClient, DashboardsServiceAsyncClient])
@mock.patch.object(DashboardsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DashboardsServiceClient))
@mock.patch.object(DashboardsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DashboardsServiceAsyncClient))
def test_dashboards_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DashboardsServiceClient, transports.DashboardsServiceGrpcTransport, 'grpc'), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DashboardsServiceClient, transports.DashboardsServiceRestTransport, 'rest')])
def test_dashboards_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DashboardsServiceClient, transports.DashboardsServiceGrpcTransport, 'grpc', grpc_helpers), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (DashboardsServiceClient, transports.DashboardsServiceRestTransport, 'rest', None)])
def test_dashboards_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_dashboards_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.monitoring_dashboard_v1.services.dashboards_service.transports.DashboardsServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DashboardsServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DashboardsServiceClient, transports.DashboardsServiceGrpcTransport, 'grpc', grpc_helpers), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_dashboards_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read', 'https://www.googleapis.com/auth/monitoring.write'), scopes=None, default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [dashboards_service.CreateDashboardRequest, dict])
def test_create_dashboard(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dashboard), '__call__') as call:
        call.return_value = dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value')
        response = client.create_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.CreateDashboardRequest()
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

def test_create_dashboard_empty_call():
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_dashboard), '__call__') as call:
        client.create_dashboard()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.CreateDashboardRequest()

@pytest.mark.asyncio
async def test_create_dashboard_async(transport: str='grpc_asyncio', request_type=dashboards_service.CreateDashboardRequest):
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value'))
        response = await client.create_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.CreateDashboardRequest()
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_create_dashboard_async_from_dict():
    await test_create_dashboard_async(request_type=dict)

def test_create_dashboard_field_headers():
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.CreateDashboardRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dashboard), '__call__') as call:
        call.return_value = dashboard.Dashboard()
        client.create_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_dashboard_field_headers_async():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.CreateDashboardRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboard.Dashboard())
        await client.create_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dashboards_service.ListDashboardsRequest, dict])
def test_list_dashboards(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        call.return_value = dashboards_service.ListDashboardsResponse(next_page_token='next_page_token_value')
        response = client.list_dashboards(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.ListDashboardsRequest()
    assert isinstance(response, pagers.ListDashboardsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_dashboards_empty_call():
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        client.list_dashboards()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.ListDashboardsRequest()

@pytest.mark.asyncio
async def test_list_dashboards_async(transport: str='grpc_asyncio', request_type=dashboards_service.ListDashboardsRequest):
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboards_service.ListDashboardsResponse(next_page_token='next_page_token_value'))
        response = await client.list_dashboards(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.ListDashboardsRequest()
    assert isinstance(response, pagers.ListDashboardsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_dashboards_async_from_dict():
    await test_list_dashboards_async(request_type=dict)

def test_list_dashboards_field_headers():
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.ListDashboardsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        call.return_value = dashboards_service.ListDashboardsResponse()
        client.list_dashboards(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_dashboards_field_headers_async():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.ListDashboardsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboards_service.ListDashboardsResponse())
        await client.list_dashboards(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_dashboards_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        call.side_effect = (dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard(), dashboard.Dashboard()], next_page_token='abc'), dashboards_service.ListDashboardsResponse(dashboards=[], next_page_token='def'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard()], next_page_token='ghi'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_dashboards(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dashboard.Dashboard) for i in results))

def test_list_dashboards_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dashboards), '__call__') as call:
        call.side_effect = (dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard(), dashboard.Dashboard()], next_page_token='abc'), dashboards_service.ListDashboardsResponse(dashboards=[], next_page_token='def'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard()], next_page_token='ghi'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard()]), RuntimeError)
        pages = list(client.list_dashboards(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_dashboards_async_pager():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dashboards), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard(), dashboard.Dashboard()], next_page_token='abc'), dashboards_service.ListDashboardsResponse(dashboards=[], next_page_token='def'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard()], next_page_token='ghi'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard()]), RuntimeError)
        async_pager = await client.list_dashboards(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dashboard.Dashboard) for i in responses))

@pytest.mark.asyncio
async def test_list_dashboards_async_pages():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dashboards), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard(), dashboard.Dashboard()], next_page_token='abc'), dashboards_service.ListDashboardsResponse(dashboards=[], next_page_token='def'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard()], next_page_token='ghi'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_dashboards(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dashboards_service.GetDashboardRequest, dict])
def test_get_dashboard(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dashboard), '__call__') as call:
        call.return_value = dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value')
        response = client.get_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.GetDashboardRequest()
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

def test_get_dashboard_empty_call():
    if False:
        return 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_dashboard), '__call__') as call:
        client.get_dashboard()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.GetDashboardRequest()

@pytest.mark.asyncio
async def test_get_dashboard_async(transport: str='grpc_asyncio', request_type=dashboards_service.GetDashboardRequest):
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value'))
        response = await client.get_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.GetDashboardRequest()
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_dashboard_async_from_dict():
    await test_get_dashboard_async(request_type=dict)

def test_get_dashboard_field_headers():
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.GetDashboardRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dashboard), '__call__') as call:
        call.return_value = dashboard.Dashboard()
        client.get_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_dashboard_field_headers_async():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.GetDashboardRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboard.Dashboard())
        await client.get_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dashboards_service.DeleteDashboardRequest, dict])
def test_delete_dashboard(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dashboard), '__call__') as call:
        call.return_value = None
        response = client.delete_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.DeleteDashboardRequest()
    assert response is None

def test_delete_dashboard_empty_call():
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_dashboard), '__call__') as call:
        client.delete_dashboard()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.DeleteDashboardRequest()

@pytest.mark.asyncio
async def test_delete_dashboard_async(transport: str='grpc_asyncio', request_type=dashboards_service.DeleteDashboardRequest):
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.DeleteDashboardRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_dashboard_async_from_dict():
    await test_delete_dashboard_async(request_type=dict)

def test_delete_dashboard_field_headers():
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.DeleteDashboardRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dashboard), '__call__') as call:
        call.return_value = None
        client.delete_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_dashboard_field_headers_async():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.DeleteDashboardRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dashboards_service.UpdateDashboardRequest, dict])
def test_update_dashboard(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dashboard), '__call__') as call:
        call.return_value = dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value')
        response = client.update_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.UpdateDashboardRequest()
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

def test_update_dashboard_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_dashboard), '__call__') as call:
        client.update_dashboard()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.UpdateDashboardRequest()

@pytest.mark.asyncio
async def test_update_dashboard_async(transport: str='grpc_asyncio', request_type=dashboards_service.UpdateDashboardRequest):
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value'))
        response = await client.update_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dashboards_service.UpdateDashboardRequest()
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_update_dashboard_async_from_dict():
    await test_update_dashboard_async(request_type=dict)

def test_update_dashboard_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.UpdateDashboardRequest()
    request.dashboard.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dashboard), '__call__') as call:
        call.return_value = dashboard.Dashboard()
        client.update_dashboard(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dashboard.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_dashboard_field_headers_async():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dashboards_service.UpdateDashboardRequest()
    request.dashboard.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dashboard), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dashboard.Dashboard())
        await client.update_dashboard(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dashboard.name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dashboards_service.CreateDashboardRequest, dict])
def test_create_dashboard_rest(request_type):
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['dashboard'] = {'name': 'name_value', 'display_name': 'display_name_value', 'etag': 'etag_value', 'grid_layout': {'columns': 769, 'widgets': [{'title': 'title_value', 'xy_chart': {'data_sets': [{'time_series_query': {'time_series_filter': {'filter': 'filter_value', 'aggregation': {'alignment_period': {'seconds': 751, 'nanos': 543}, 'per_series_aligner': 1, 'cross_series_reducer': 1, 'group_by_fields': ['group_by_fields_value1', 'group_by_fields_value2']}, 'secondary_aggregation': {}, 'pick_time_series_filter': {'ranking_method': 1, 'num_time_series': 1608, 'direction': 1}, 'statistical_time_series_filter': {'ranking_method': 1, 'num_time_series': 1608}}, 'time_series_filter_ratio': {'numerator': {'filter': 'filter_value', 'aggregation': {}}, 'denominator': {}, 'secondary_aggregation': {}, 'pick_time_series_filter': {}, 'statistical_time_series_filter': {}}, 'time_series_query_language': 'time_series_query_language_value', 'prometheus_query': 'prometheus_query_value', 'unit_override': 'unit_override_value'}, 'plot_type': 1, 'legend_template': 'legend_template_value', 'min_alignment_period': {}, 'target_axis': 1}], 'timeshift_duration': {}, 'thresholds': [{'label': 'label_value', 'value': 0.541, 'color': 4, 'direction': 1, 'target_axis': 1}], 'x_axis': {'label': 'label_value', 'scale': 1}, 'y_axis': {}, 'y2_axis': {}, 'chart_options': {'mode': 1}}, 'scorecard': {'time_series_query': {}, 'gauge_view': {'lower_bound': 0.1184, 'upper_bound': 0.1187}, 'spark_chart_view': {'spark_chart_type': 1, 'min_alignment_period': {}}, 'thresholds': {}}, 'text': {'content': 'content_value', 'format_': 1}, 'blank': {}, 'alert_chart': {'name': 'name_value'}, 'time_series_table': {'data_sets': [{'time_series_query': {}, 'table_template': 'table_template_value', 'min_alignment_period': {}, 'table_display_options': {'shown_columns': ['shown_columns_value1', 'shown_columns_value2']}}], 'metric_visualization': 1, 'column_settings': [{'column': 'column_value', 'visible': True}]}, 'collapsible_group': {'collapsed': True}, 'logs_panel': {'filter': 'filter_value', 'resource_names': ['resource_names_value1', 'resource_names_value2']}}]}, 'mosaic_layout': {'columns': 769, 'tiles': [{'x_pos': 553, 'y_pos': 554, 'width': 544, 'height': 633, 'widget': {}}]}, 'row_layout': {'rows': [{'weight': 648, 'widgets': {}}]}, 'column_layout': {'columns': [{'weight': 648, 'widgets': {}}]}, 'dashboard_filters': [{'label_key': 'label_key_value', 'template_variable': 'template_variable_value', 'string_value': 'string_value_value', 'filter_type': 1}], 'labels': {}}
    test_field = dashboards_service.CreateDashboardRequest.meta.fields['dashboard']

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
    for (field, value) in request_init['dashboard'].items():
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
                for i in range(0, len(request_init['dashboard'][field])):
                    del request_init['dashboard'][field][i][subfield]
            else:
                del request_init['dashboard'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dashboard.Dashboard.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_dashboard(request)
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

def test_create_dashboard_rest_required_fields(request_type=dashboards_service.CreateDashboardRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DashboardsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dashboard._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dashboard._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dashboard.Dashboard()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dashboard.Dashboard.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_dashboard(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_dashboard_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_dashboard._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('parent', 'dashboard'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_dashboard_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DashboardsServiceRestInterceptor())
    client = DashboardsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'post_create_dashboard') as post, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'pre_create_dashboard') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dashboards_service.CreateDashboardRequest.pb(dashboards_service.CreateDashboardRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dashboard.Dashboard.to_json(dashboard.Dashboard())
        request = dashboards_service.CreateDashboardRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dashboard.Dashboard()
        client.create_dashboard(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_dashboard_rest_bad_request(transport: str='rest', request_type=dashboards_service.CreateDashboardRequest):
    if False:
        return 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_dashboard(request)

def test_create_dashboard_rest_error():
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dashboards_service.ListDashboardsRequest, dict])
def test_list_dashboards_rest(request_type):
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dashboards_service.ListDashboardsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dashboards_service.ListDashboardsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_dashboards(request)
    assert isinstance(response, pagers.ListDashboardsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_dashboards_rest_required_fields(request_type=dashboards_service.ListDashboardsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DashboardsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dashboards._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dashboards._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dashboards_service.ListDashboardsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dashboards_service.ListDashboardsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_dashboards(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_dashboards_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_dashboards._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_dashboards_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DashboardsServiceRestInterceptor())
    client = DashboardsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'post_list_dashboards') as post, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'pre_list_dashboards') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dashboards_service.ListDashboardsRequest.pb(dashboards_service.ListDashboardsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dashboards_service.ListDashboardsResponse.to_json(dashboards_service.ListDashboardsResponse())
        request = dashboards_service.ListDashboardsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dashboards_service.ListDashboardsResponse()
        client.list_dashboards(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_dashboards_rest_bad_request(transport: str='rest', request_type=dashboards_service.ListDashboardsRequest):
    if False:
        return 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_dashboards(request)

def test_list_dashboards_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard(), dashboard.Dashboard()], next_page_token='abc'), dashboards_service.ListDashboardsResponse(dashboards=[], next_page_token='def'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard()], next_page_token='ghi'), dashboards_service.ListDashboardsResponse(dashboards=[dashboard.Dashboard(), dashboard.Dashboard()]))
        response = response + response
        response = tuple((dashboards_service.ListDashboardsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_dashboards(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dashboard.Dashboard) for i in results))
        pages = list(client.list_dashboards(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dashboards_service.GetDashboardRequest, dict])
def test_get_dashboard_rest(request_type):
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/dashboards/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dashboard.Dashboard.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_dashboard(request)
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

def test_get_dashboard_rest_required_fields(request_type=dashboards_service.GetDashboardRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DashboardsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dashboard._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dashboard._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dashboard.Dashboard()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dashboard.Dashboard.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_dashboard(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_dashboard_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_dashboard._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_dashboard_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DashboardsServiceRestInterceptor())
    client = DashboardsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'post_get_dashboard') as post, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'pre_get_dashboard') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dashboards_service.GetDashboardRequest.pb(dashboards_service.GetDashboardRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dashboard.Dashboard.to_json(dashboard.Dashboard())
        request = dashboards_service.GetDashboardRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dashboard.Dashboard()
        client.get_dashboard(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_dashboard_rest_bad_request(transport: str='rest', request_type=dashboards_service.GetDashboardRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/dashboards/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_dashboard(request)

def test_get_dashboard_rest_error():
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dashboards_service.DeleteDashboardRequest, dict])
def test_delete_dashboard_rest(request_type):
    if False:
        print('Hello World!')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/dashboards/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_dashboard(request)
    assert response is None

def test_delete_dashboard_rest_required_fields(request_type=dashboards_service.DeleteDashboardRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DashboardsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dashboard._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dashboard._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_dashboard(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_dashboard_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_dashboard._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_dashboard_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DashboardsServiceRestInterceptor())
    client = DashboardsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'pre_delete_dashboard') as pre:
        pre.assert_not_called()
        pb_message = dashboards_service.DeleteDashboardRequest.pb(dashboards_service.DeleteDashboardRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dashboards_service.DeleteDashboardRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_dashboard(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_dashboard_rest_bad_request(transport: str='rest', request_type=dashboards_service.DeleteDashboardRequest):
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/dashboards/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_dashboard(request)

def test_delete_dashboard_rest_error():
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dashboards_service.UpdateDashboardRequest, dict])
def test_update_dashboard_rest(request_type):
    if False:
        return 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dashboard': {'name': 'projects/sample1/dashboards/sample2'}}
    request_init['dashboard'] = {'name': 'projects/sample1/dashboards/sample2', 'display_name': 'display_name_value', 'etag': 'etag_value', 'grid_layout': {'columns': 769, 'widgets': [{'title': 'title_value', 'xy_chart': {'data_sets': [{'time_series_query': {'time_series_filter': {'filter': 'filter_value', 'aggregation': {'alignment_period': {'seconds': 751, 'nanos': 543}, 'per_series_aligner': 1, 'cross_series_reducer': 1, 'group_by_fields': ['group_by_fields_value1', 'group_by_fields_value2']}, 'secondary_aggregation': {}, 'pick_time_series_filter': {'ranking_method': 1, 'num_time_series': 1608, 'direction': 1}, 'statistical_time_series_filter': {'ranking_method': 1, 'num_time_series': 1608}}, 'time_series_filter_ratio': {'numerator': {'filter': 'filter_value', 'aggregation': {}}, 'denominator': {}, 'secondary_aggregation': {}, 'pick_time_series_filter': {}, 'statistical_time_series_filter': {}}, 'time_series_query_language': 'time_series_query_language_value', 'prometheus_query': 'prometheus_query_value', 'unit_override': 'unit_override_value'}, 'plot_type': 1, 'legend_template': 'legend_template_value', 'min_alignment_period': {}, 'target_axis': 1}], 'timeshift_duration': {}, 'thresholds': [{'label': 'label_value', 'value': 0.541, 'color': 4, 'direction': 1, 'target_axis': 1}], 'x_axis': {'label': 'label_value', 'scale': 1}, 'y_axis': {}, 'y2_axis': {}, 'chart_options': {'mode': 1}}, 'scorecard': {'time_series_query': {}, 'gauge_view': {'lower_bound': 0.1184, 'upper_bound': 0.1187}, 'spark_chart_view': {'spark_chart_type': 1, 'min_alignment_period': {}}, 'thresholds': {}}, 'text': {'content': 'content_value', 'format_': 1}, 'blank': {}, 'alert_chart': {'name': 'name_value'}, 'time_series_table': {'data_sets': [{'time_series_query': {}, 'table_template': 'table_template_value', 'min_alignment_period': {}, 'table_display_options': {'shown_columns': ['shown_columns_value1', 'shown_columns_value2']}}], 'metric_visualization': 1, 'column_settings': [{'column': 'column_value', 'visible': True}]}, 'collapsible_group': {'collapsed': True}, 'logs_panel': {'filter': 'filter_value', 'resource_names': ['resource_names_value1', 'resource_names_value2']}}]}, 'mosaic_layout': {'columns': 769, 'tiles': [{'x_pos': 553, 'y_pos': 554, 'width': 544, 'height': 633, 'widget': {}}]}, 'row_layout': {'rows': [{'weight': 648, 'widgets': {}}]}, 'column_layout': {'columns': [{'weight': 648, 'widgets': {}}]}, 'dashboard_filters': [{'label_key': 'label_key_value', 'template_variable': 'template_variable_value', 'string_value': 'string_value_value', 'filter_type': 1}], 'labels': {}}
    test_field = dashboards_service.UpdateDashboardRequest.meta.fields['dashboard']

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
    for (field, value) in request_init['dashboard'].items():
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
                for i in range(0, len(request_init['dashboard'][field])):
                    del request_init['dashboard'][field][i][subfield]
            else:
                del request_init['dashboard'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dashboard.Dashboard(name='name_value', display_name='display_name_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dashboard.Dashboard.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_dashboard(request)
    assert isinstance(response, dashboard.Dashboard)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.etag == 'etag_value'

def test_update_dashboard_rest_required_fields(request_type=dashboards_service.UpdateDashboardRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DashboardsServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dashboard._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dashboard._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dashboard.Dashboard()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dashboard.Dashboard.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_dashboard(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_dashboard_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_dashboard._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('dashboard',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_dashboard_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DashboardsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DashboardsServiceRestInterceptor())
    client = DashboardsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'post_update_dashboard') as post, mock.patch.object(transports.DashboardsServiceRestInterceptor, 'pre_update_dashboard') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dashboards_service.UpdateDashboardRequest.pb(dashboards_service.UpdateDashboardRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dashboard.Dashboard.to_json(dashboard.Dashboard())
        request = dashboards_service.UpdateDashboardRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dashboard.Dashboard()
        client.update_dashboard(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_dashboard_rest_bad_request(transport: str='rest', request_type=dashboards_service.UpdateDashboardRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dashboard': {'name': 'projects/sample1/dashboards/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_dashboard(request)

def test_update_dashboard_rest_error():
    if False:
        i = 10
        return i + 15
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DashboardsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DashboardsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DashboardsServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DashboardsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DashboardsServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DashboardsServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DashboardsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DashboardsServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.DashboardsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DashboardsServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DashboardsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DashboardsServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceGrpcAsyncIOTransport, transports.DashboardsServiceRestTransport])
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
        i = 10
        return i + 15
    transport = DashboardsServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DashboardsServiceGrpcTransport)

def test_dashboards_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DashboardsServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_dashboards_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.monitoring_dashboard_v1.services.dashboards_service.transports.DashboardsServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DashboardsServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_dashboard', 'list_dashboards', 'get_dashboard', 'delete_dashboard', 'update_dashboard')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_dashboards_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.monitoring_dashboard_v1.services.dashboards_service.transports.DashboardsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DashboardsServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read', 'https://www.googleapis.com/auth/monitoring.write'), quota_project_id='octopus')

def test_dashboards_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.monitoring_dashboard_v1.services.dashboards_service.transports.DashboardsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DashboardsServiceTransport()
        adc.assert_called_once()

def test_dashboards_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DashboardsServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read', 'https://www.googleapis.com/auth/monitoring.write'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceGrpcAsyncIOTransport])
def test_dashboards_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read', 'https://www.googleapis.com/auth/monitoring.write'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceGrpcAsyncIOTransport, transports.DashboardsServiceRestTransport])
def test_dashboards_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DashboardsServiceGrpcTransport, grpc_helpers), (transports.DashboardsServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_dashboards_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read', 'https://www.googleapis.com/auth/monitoring.write'), scopes=['1', '2'], default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceGrpcAsyncIOTransport])
def test_dashboards_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_dashboards_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.DashboardsServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_dashboards_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('monitoring.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://monitoring.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_dashboards_service_host_with_port(transport_name):
    if False:
        return 10
    client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('monitoring.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://monitoring.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_dashboards_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = DashboardsServiceClient(credentials=creds1, transport=transport_name)
    client2 = DashboardsServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_dashboard._session
    session2 = client2.transport.create_dashboard._session
    assert session1 != session2
    session1 = client1.transport.list_dashboards._session
    session2 = client2.transport.list_dashboards._session
    assert session1 != session2
    session1 = client1.transport.get_dashboard._session
    session2 = client2.transport.get_dashboard._session
    assert session1 != session2
    session1 = client1.transport.delete_dashboard._session
    session2 = client2.transport.delete_dashboard._session
    assert session1 != session2
    session1 = client1.transport.update_dashboard._session
    session2 = client2.transport.update_dashboard._session
    assert session1 != session2

def test_dashboards_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DashboardsServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_dashboards_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DashboardsServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceGrpcAsyncIOTransport])
def test_dashboards_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.DashboardsServiceGrpcTransport, transports.DashboardsServiceGrpcAsyncIOTransport])
def test_dashboards_service_transport_channel_mtls_with_adc(transport_class):
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

def test_alert_policy_path():
    if False:
        print('Hello World!')
    project = 'squid'
    alert_policy = 'clam'
    expected = 'projects/{project}/alertPolicies/{alert_policy}'.format(project=project, alert_policy=alert_policy)
    actual = DashboardsServiceClient.alert_policy_path(project, alert_policy)
    assert expected == actual

def test_parse_alert_policy_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'alert_policy': 'octopus'}
    path = DashboardsServiceClient.alert_policy_path(**expected)
    actual = DashboardsServiceClient.parse_alert_policy_path(path)
    assert expected == actual

def test_dashboard_path():
    if False:
        return 10
    project = 'oyster'
    dashboard = 'nudibranch'
    expected = 'projects/{project}/dashboards/{dashboard}'.format(project=project, dashboard=dashboard)
    actual = DashboardsServiceClient.dashboard_path(project, dashboard)
    assert expected == actual

def test_parse_dashboard_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'cuttlefish', 'dashboard': 'mussel'}
    path = DashboardsServiceClient.dashboard_path(**expected)
    actual = DashboardsServiceClient.parse_dashboard_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DashboardsServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'nautilus'}
    path = DashboardsServiceClient.common_billing_account_path(**expected)
    actual = DashboardsServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DashboardsServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'abalone'}
    path = DashboardsServiceClient.common_folder_path(**expected)
    actual = DashboardsServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DashboardsServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'clam'}
    path = DashboardsServiceClient.common_organization_path(**expected)
    actual = DashboardsServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = DashboardsServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus'}
    path = DashboardsServiceClient.common_project_path(**expected)
    actual = DashboardsServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DashboardsServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = DashboardsServiceClient.common_location_path(**expected)
    actual = DashboardsServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DashboardsServiceTransport, '_prep_wrapped_messages') as prep:
        client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DashboardsServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DashboardsServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DashboardsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = DashboardsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DashboardsServiceClient, transports.DashboardsServiceGrpcTransport), (DashboardsServiceAsyncClient, transports.DashboardsServiceGrpcAsyncIOTransport)])
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