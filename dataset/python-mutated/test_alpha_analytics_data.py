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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.analytics.data_v1alpha.services.alpha_analytics_data import AlphaAnalyticsDataAsyncClient, AlphaAnalyticsDataClient, pagers, transports
from google.analytics.data_v1alpha.types import analytics_data_api, data

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
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
    assert AlphaAnalyticsDataClient._get_default_mtls_endpoint(None) is None
    assert AlphaAnalyticsDataClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AlphaAnalyticsDataClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AlphaAnalyticsDataClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AlphaAnalyticsDataClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AlphaAnalyticsDataClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AlphaAnalyticsDataClient, 'grpc'), (AlphaAnalyticsDataAsyncClient, 'grpc_asyncio'), (AlphaAnalyticsDataClient, 'rest')])
def test_alpha_analytics_data_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('analyticsdata.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://analyticsdata.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AlphaAnalyticsDataGrpcTransport, 'grpc'), (transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AlphaAnalyticsDataRestTransport, 'rest')])
def test_alpha_analytics_data_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AlphaAnalyticsDataClient, 'grpc'), (AlphaAnalyticsDataAsyncClient, 'grpc_asyncio'), (AlphaAnalyticsDataClient, 'rest')])
def test_alpha_analytics_data_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('analyticsdata.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://analyticsdata.googleapis.com')

def test_alpha_analytics_data_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = AlphaAnalyticsDataClient.get_transport_class()
    available_transports = [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataRestTransport]
    assert transport in available_transports
    transport = AlphaAnalyticsDataClient.get_transport_class('grpc')
    assert transport == transports.AlphaAnalyticsDataGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport, 'grpc'), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio'), (AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataRestTransport, 'rest')])
@mock.patch.object(AlphaAnalyticsDataClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlphaAnalyticsDataClient))
@mock.patch.object(AlphaAnalyticsDataAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlphaAnalyticsDataAsyncClient))
def test_alpha_analytics_data_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(AlphaAnalyticsDataClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AlphaAnalyticsDataClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport, 'grpc', 'true'), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport, 'grpc', 'false'), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataRestTransport, 'rest', 'true'), (AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataRestTransport, 'rest', 'false')])
@mock.patch.object(AlphaAnalyticsDataClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlphaAnalyticsDataClient))
@mock.patch.object(AlphaAnalyticsDataAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlphaAnalyticsDataAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_alpha_analytics_data_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AlphaAnalyticsDataClient, AlphaAnalyticsDataAsyncClient])
@mock.patch.object(AlphaAnalyticsDataClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlphaAnalyticsDataClient))
@mock.patch.object(AlphaAnalyticsDataAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlphaAnalyticsDataAsyncClient))
def test_alpha_analytics_data_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport, 'grpc'), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio'), (AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataRestTransport, 'rest')])
def test_alpha_analytics_data_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport, 'grpc', grpc_helpers), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataRestTransport, 'rest', None)])
def test_alpha_analytics_data_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_alpha_analytics_data_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.analytics.data_v1alpha.services.alpha_analytics_data.transports.AlphaAnalyticsDataGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AlphaAnalyticsDataClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport, 'grpc', grpc_helpers), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_alpha_analytics_data_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('analyticsdata.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/analytics', 'https://www.googleapis.com/auth/analytics.readonly', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/spreadsheets'), scopes=None, default_host='analyticsdata.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [analytics_data_api.RunFunnelReportRequest, dict])
def test_run_funnel_report(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_funnel_report), '__call__') as call:
        call.return_value = analytics_data_api.RunFunnelReportResponse(kind='kind_value')
        response = client.run_funnel_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.RunFunnelReportRequest()
    assert isinstance(response, analytics_data_api.RunFunnelReportResponse)
    assert response.kind == 'kind_value'

def test_run_funnel_report_empty_call():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.run_funnel_report), '__call__') as call:
        client.run_funnel_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.RunFunnelReportRequest()

@pytest.mark.asyncio
async def test_run_funnel_report_async(transport: str='grpc_asyncio', request_type=analytics_data_api.RunFunnelReportRequest):
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_funnel_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.RunFunnelReportResponse(kind='kind_value'))
        response = await client.run_funnel_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.RunFunnelReportRequest()
    assert isinstance(response, analytics_data_api.RunFunnelReportResponse)
    assert response.kind == 'kind_value'

@pytest.mark.asyncio
async def test_run_funnel_report_async_from_dict():
    await test_run_funnel_report_async(request_type=dict)

def test_run_funnel_report_field_headers():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.RunFunnelReportRequest()
    request.property = 'property_value'
    with mock.patch.object(type(client.transport.run_funnel_report), '__call__') as call:
        call.return_value = analytics_data_api.RunFunnelReportResponse()
        client.run_funnel_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'property=property_value') in kw['metadata']

@pytest.mark.asyncio
async def test_run_funnel_report_field_headers_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.RunFunnelReportRequest()
    request.property = 'property_value'
    with mock.patch.object(type(client.transport.run_funnel_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.RunFunnelReportResponse())
        await client.run_funnel_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'property=property_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [analytics_data_api.CreateAudienceListRequest, dict])
def test_create_audience_list(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.CreateAudienceListRequest()
    assert isinstance(response, future.Future)

def test_create_audience_list_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        client.create_audience_list()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.CreateAudienceListRequest()

@pytest.mark.asyncio
async def test_create_audience_list_async(transport: str='grpc_asyncio', request_type=analytics_data_api.CreateAudienceListRequest):
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.CreateAudienceListRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_audience_list_async_from_dict():
    await test_create_audience_list_async(request_type=dict)

def test_create_audience_list_field_headers():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.CreateAudienceListRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_audience_list_field_headers_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.CreateAudienceListRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_audience_list_flattened():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_audience_list(parent='parent_value', audience_list=analytics_data_api.AudienceList(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].audience_list
        mock_val = analytics_data_api.AudienceList(name='name_value')
        assert arg == mock_val

def test_create_audience_list_flattened_error():
    if False:
        while True:
            i = 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_audience_list(analytics_data_api.CreateAudienceListRequest(), parent='parent_value', audience_list=analytics_data_api.AudienceList(name='name_value'))

@pytest.mark.asyncio
async def test_create_audience_list_flattened_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_audience_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_audience_list(parent='parent_value', audience_list=analytics_data_api.AudienceList(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].audience_list
        mock_val = analytics_data_api.AudienceList(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_audience_list_flattened_error_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_audience_list(analytics_data_api.CreateAudienceListRequest(), parent='parent_value', audience_list=analytics_data_api.AudienceList(name='name_value'))

@pytest.mark.parametrize('request_type', [analytics_data_api.QueryAudienceListRequest, dict])
def test_query_audience_list(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.QueryAudienceListResponse(row_count=992)
        response = client.query_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.QueryAudienceListRequest()
    assert isinstance(response, analytics_data_api.QueryAudienceListResponse)
    assert response.row_count == 992

def test_query_audience_list_empty_call():
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        client.query_audience_list()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.QueryAudienceListRequest()

@pytest.mark.asyncio
async def test_query_audience_list_async(transport: str='grpc_asyncio', request_type=analytics_data_api.QueryAudienceListRequest):
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.QueryAudienceListResponse(row_count=992))
        response = await client.query_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.QueryAudienceListRequest()
    assert isinstance(response, analytics_data_api.QueryAudienceListResponse)
    assert response.row_count == 992

@pytest.mark.asyncio
async def test_query_audience_list_async_from_dict():
    await test_query_audience_list_async(request_type=dict)

def test_query_audience_list_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.QueryAudienceListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.QueryAudienceListResponse()
        client.query_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_query_audience_list_field_headers_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.QueryAudienceListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.QueryAudienceListResponse())
        await client.query_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_query_audience_list_flattened():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.QueryAudienceListResponse()
        client.query_audience_list(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_query_audience_list_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.query_audience_list(analytics_data_api.QueryAudienceListRequest(), name='name_value')

@pytest.mark.asyncio
async def test_query_audience_list_flattened_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.query_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.QueryAudienceListResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.QueryAudienceListResponse())
        response = await client.query_audience_list(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_query_audience_list_flattened_error_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.query_audience_list(analytics_data_api.QueryAudienceListRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [analytics_data_api.SheetExportAudienceListRequest, dict])
def test_sheet_export_audience_list(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.SheetExportAudienceListResponse(spreadsheet_uri='spreadsheet_uri_value', spreadsheet_id='spreadsheet_id_value', row_count=992)
        response = client.sheet_export_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.SheetExportAudienceListRequest()
    assert isinstance(response, analytics_data_api.SheetExportAudienceListResponse)
    assert response.spreadsheet_uri == 'spreadsheet_uri_value'
    assert response.spreadsheet_id == 'spreadsheet_id_value'
    assert response.row_count == 992

def test_sheet_export_audience_list_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        client.sheet_export_audience_list()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.SheetExportAudienceListRequest()

@pytest.mark.asyncio
async def test_sheet_export_audience_list_async(transport: str='grpc_asyncio', request_type=analytics_data_api.SheetExportAudienceListRequest):
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.SheetExportAudienceListResponse(spreadsheet_uri='spreadsheet_uri_value', spreadsheet_id='spreadsheet_id_value', row_count=992))
        response = await client.sheet_export_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.SheetExportAudienceListRequest()
    assert isinstance(response, analytics_data_api.SheetExportAudienceListResponse)
    assert response.spreadsheet_uri == 'spreadsheet_uri_value'
    assert response.spreadsheet_id == 'spreadsheet_id_value'
    assert response.row_count == 992

@pytest.mark.asyncio
async def test_sheet_export_audience_list_async_from_dict():
    await test_sheet_export_audience_list_async(request_type=dict)

def test_sheet_export_audience_list_field_headers():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.SheetExportAudienceListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.SheetExportAudienceListResponse()
        client.sheet_export_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_sheet_export_audience_list_field_headers_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.SheetExportAudienceListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.SheetExportAudienceListResponse())
        await client.sheet_export_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_sheet_export_audience_list_flattened():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.SheetExportAudienceListResponse()
        client.sheet_export_audience_list(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_sheet_export_audience_list_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.sheet_export_audience_list(analytics_data_api.SheetExportAudienceListRequest(), name='name_value')

@pytest.mark.asyncio
async def test_sheet_export_audience_list_flattened_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.sheet_export_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.SheetExportAudienceListResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.SheetExportAudienceListResponse())
        response = await client.sheet_export_audience_list(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_sheet_export_audience_list_flattened_error_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.sheet_export_audience_list(analytics_data_api.SheetExportAudienceListRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [analytics_data_api.GetAudienceListRequest, dict])
def test_get_audience_list(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.AudienceList(name='name_value', audience='audience_value', audience_display_name='audience_display_name_value', state=analytics_data_api.AudienceList.State.CREATING, creation_quota_tokens_charged=3070, row_count=992, error_message='error_message_value')
        response = client.get_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.GetAudienceListRequest()
    assert isinstance(response, analytics_data_api.AudienceList)
    assert response.name == 'name_value'
    assert response.audience == 'audience_value'
    assert response.audience_display_name == 'audience_display_name_value'
    assert response.state == analytics_data_api.AudienceList.State.CREATING
    assert response.creation_quota_tokens_charged == 3070
    assert response.row_count == 992
    assert response.error_message == 'error_message_value'

def test_get_audience_list_empty_call():
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        client.get_audience_list()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.GetAudienceListRequest()

@pytest.mark.asyncio
async def test_get_audience_list_async(transport: str='grpc_asyncio', request_type=analytics_data_api.GetAudienceListRequest):
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.AudienceList(name='name_value', audience='audience_value', audience_display_name='audience_display_name_value', state=analytics_data_api.AudienceList.State.CREATING, creation_quota_tokens_charged=3070, row_count=992, error_message='error_message_value'))
        response = await client.get_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.GetAudienceListRequest()
    assert isinstance(response, analytics_data_api.AudienceList)
    assert response.name == 'name_value'
    assert response.audience == 'audience_value'
    assert response.audience_display_name == 'audience_display_name_value'
    assert response.state == analytics_data_api.AudienceList.State.CREATING
    assert response.creation_quota_tokens_charged == 3070
    assert response.row_count == 992
    assert response.error_message == 'error_message_value'

@pytest.mark.asyncio
async def test_get_audience_list_async_from_dict():
    await test_get_audience_list_async(request_type=dict)

def test_get_audience_list_field_headers():
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.GetAudienceListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.AudienceList()
        client.get_audience_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_audience_list_field_headers_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.GetAudienceListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.AudienceList())
        await client.get_audience_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_audience_list_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.AudienceList()
        client.get_audience_list(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_audience_list_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_audience_list(analytics_data_api.GetAudienceListRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_audience_list_flattened_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_audience_list), '__call__') as call:
        call.return_value = analytics_data_api.AudienceList()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.AudienceList())
        response = await client.get_audience_list(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_audience_list_flattened_error_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_audience_list(analytics_data_api.GetAudienceListRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [analytics_data_api.ListAudienceListsRequest, dict])
def test_list_audience_lists(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.return_value = analytics_data_api.ListAudienceListsResponse(next_page_token='next_page_token_value')
        response = client.list_audience_lists(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.ListAudienceListsRequest()
    assert isinstance(response, pagers.ListAudienceListsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_audience_lists_empty_call():
    if False:
        while True:
            i = 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        client.list_audience_lists()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.ListAudienceListsRequest()

@pytest.mark.asyncio
async def test_list_audience_lists_async(transport: str='grpc_asyncio', request_type=analytics_data_api.ListAudienceListsRequest):
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.ListAudienceListsResponse(next_page_token='next_page_token_value'))
        response = await client.list_audience_lists(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == analytics_data_api.ListAudienceListsRequest()
    assert isinstance(response, pagers.ListAudienceListsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_audience_lists_async_from_dict():
    await test_list_audience_lists_async(request_type=dict)

def test_list_audience_lists_field_headers():
    if False:
        while True:
            i = 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.ListAudienceListsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.return_value = analytics_data_api.ListAudienceListsResponse()
        client.list_audience_lists(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_audience_lists_field_headers_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = analytics_data_api.ListAudienceListsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.ListAudienceListsResponse())
        await client.list_audience_lists(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_audience_lists_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.return_value = analytics_data_api.ListAudienceListsResponse()
        client.list_audience_lists(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_audience_lists_flattened_error():
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_audience_lists(analytics_data_api.ListAudienceListsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_audience_lists_flattened_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.return_value = analytics_data_api.ListAudienceListsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(analytics_data_api.ListAudienceListsResponse())
        response = await client.list_audience_lists(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_audience_lists_flattened_error_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_audience_lists(analytics_data_api.ListAudienceListsRequest(), parent='parent_value')

def test_list_audience_lists_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.side_effect = (analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList(), analytics_data_api.AudienceList()], next_page_token='abc'), analytics_data_api.ListAudienceListsResponse(audience_lists=[], next_page_token='def'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList()], next_page_token='ghi'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_audience_lists(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, analytics_data_api.AudienceList) for i in results))

def test_list_audience_lists_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__') as call:
        call.side_effect = (analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList(), analytics_data_api.AudienceList()], next_page_token='abc'), analytics_data_api.ListAudienceListsResponse(audience_lists=[], next_page_token='def'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList()], next_page_token='ghi'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList()]), RuntimeError)
        pages = list(client.list_audience_lists(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_audience_lists_async_pager():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList(), analytics_data_api.AudienceList()], next_page_token='abc'), analytics_data_api.ListAudienceListsResponse(audience_lists=[], next_page_token='def'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList()], next_page_token='ghi'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList()]), RuntimeError)
        async_pager = await client.list_audience_lists(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, analytics_data_api.AudienceList) for i in responses))

@pytest.mark.asyncio
async def test_list_audience_lists_async_pages():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_audience_lists), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList(), analytics_data_api.AudienceList()], next_page_token='abc'), analytics_data_api.ListAudienceListsResponse(audience_lists=[], next_page_token='def'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList()], next_page_token='ghi'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_audience_lists(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [analytics_data_api.RunFunnelReportRequest, dict])
def test_run_funnel_report_rest(request_type):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'property': 'properties/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.RunFunnelReportResponse(kind='kind_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.RunFunnelReportResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.run_funnel_report(request)
    assert isinstance(response, analytics_data_api.RunFunnelReportResponse)
    assert response.kind == 'kind_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_run_funnel_report_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlphaAnalyticsDataRestInterceptor())
    client = AlphaAnalyticsDataClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'post_run_funnel_report') as post, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'pre_run_funnel_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = analytics_data_api.RunFunnelReportRequest.pb(analytics_data_api.RunFunnelReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = analytics_data_api.RunFunnelReportResponse.to_json(analytics_data_api.RunFunnelReportResponse())
        request = analytics_data_api.RunFunnelReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = analytics_data_api.RunFunnelReportResponse()
        client.run_funnel_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_run_funnel_report_rest_bad_request(transport: str='rest', request_type=analytics_data_api.RunFunnelReportRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'property': 'properties/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.run_funnel_report(request)

def test_run_funnel_report_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [analytics_data_api.CreateAudienceListRequest, dict])
def test_create_audience_list_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'properties/sample1'}
    request_init['audience_list'] = {'name': 'name_value', 'audience': 'audience_value', 'audience_display_name': 'audience_display_name_value', 'dimensions': [{'dimension_name': 'dimension_name_value'}], 'state': 1, 'begin_creating_time': {'seconds': 751, 'nanos': 543}, 'creation_quota_tokens_charged': 3070, 'row_count': 992, 'error_message': 'error_message_value'}
    test_field = analytics_data_api.CreateAudienceListRequest.meta.fields['audience_list']

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
    for (field, value) in request_init['audience_list'].items():
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
                for i in range(0, len(request_init['audience_list'][field])):
                    del request_init['audience_list'][field][i][subfield]
            else:
                del request_init['audience_list'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_audience_list(request)
    assert response.operation.name == 'operations/spam'

def test_create_audience_list_rest_required_fields(request_type=analytics_data_api.CreateAudienceListRequest):
    if False:
        return 10
    transport_class = transports.AlphaAnalyticsDataRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_audience_list(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_audience_list_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_audience_list._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'audienceList'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_audience_list_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlphaAnalyticsDataRestInterceptor())
    client = AlphaAnalyticsDataClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'post_create_audience_list') as post, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'pre_create_audience_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = analytics_data_api.CreateAudienceListRequest.pb(analytics_data_api.CreateAudienceListRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = analytics_data_api.CreateAudienceListRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_audience_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_audience_list_rest_bad_request(transport: str='rest', request_type=analytics_data_api.CreateAudienceListRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'properties/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_audience_list(request)

def test_create_audience_list_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'properties/sample1'}
        mock_args = dict(parent='parent_value', audience_list=analytics_data_api.AudienceList(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_audience_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=properties/*}/audienceLists' % client.transport._host, args[1])

def test_create_audience_list_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_audience_list(analytics_data_api.CreateAudienceListRequest(), parent='parent_value', audience_list=analytics_data_api.AudienceList(name='name_value'))

def test_create_audience_list_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [analytics_data_api.QueryAudienceListRequest, dict])
def test_query_audience_list_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'properties/sample1/audienceLists/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.QueryAudienceListResponse(row_count=992)
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.QueryAudienceListResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.query_audience_list(request)
    assert isinstance(response, analytics_data_api.QueryAudienceListResponse)
    assert response.row_count == 992

def test_query_audience_list_rest_required_fields(request_type=analytics_data_api.QueryAudienceListRequest):
    if False:
        return 10
    transport_class = transports.AlphaAnalyticsDataRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).query_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).query_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = analytics_data_api.QueryAudienceListResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = analytics_data_api.QueryAudienceListResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.query_audience_list(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_query_audience_list_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.query_audience_list._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_query_audience_list_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlphaAnalyticsDataRestInterceptor())
    client = AlphaAnalyticsDataClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'post_query_audience_list') as post, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'pre_query_audience_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = analytics_data_api.QueryAudienceListRequest.pb(analytics_data_api.QueryAudienceListRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = analytics_data_api.QueryAudienceListResponse.to_json(analytics_data_api.QueryAudienceListResponse())
        request = analytics_data_api.QueryAudienceListRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = analytics_data_api.QueryAudienceListResponse()
        client.query_audience_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_query_audience_list_rest_bad_request(transport: str='rest', request_type=analytics_data_api.QueryAudienceListRequest):
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'properties/sample1/audienceLists/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.query_audience_list(request)

def test_query_audience_list_rest_flattened():
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.QueryAudienceListResponse()
        sample_request = {'name': 'properties/sample1/audienceLists/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.QueryAudienceListResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.query_audience_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=properties/*/audienceLists/*}:query' % client.transport._host, args[1])

def test_query_audience_list_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.query_audience_list(analytics_data_api.QueryAudienceListRequest(), name='name_value')

def test_query_audience_list_rest_error():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [analytics_data_api.SheetExportAudienceListRequest, dict])
def test_sheet_export_audience_list_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'properties/sample1/audienceLists/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.SheetExportAudienceListResponse(spreadsheet_uri='spreadsheet_uri_value', spreadsheet_id='spreadsheet_id_value', row_count=992)
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.SheetExportAudienceListResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.sheet_export_audience_list(request)
    assert isinstance(response, analytics_data_api.SheetExportAudienceListResponse)
    assert response.spreadsheet_uri == 'spreadsheet_uri_value'
    assert response.spreadsheet_id == 'spreadsheet_id_value'
    assert response.row_count == 992

def test_sheet_export_audience_list_rest_required_fields(request_type=analytics_data_api.SheetExportAudienceListRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlphaAnalyticsDataRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).sheet_export_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).sheet_export_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = analytics_data_api.SheetExportAudienceListResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = analytics_data_api.SheetExportAudienceListResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.sheet_export_audience_list(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_sheet_export_audience_list_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.sheet_export_audience_list._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_sheet_export_audience_list_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlphaAnalyticsDataRestInterceptor())
    client = AlphaAnalyticsDataClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'post_sheet_export_audience_list') as post, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'pre_sheet_export_audience_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = analytics_data_api.SheetExportAudienceListRequest.pb(analytics_data_api.SheetExportAudienceListRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = analytics_data_api.SheetExportAudienceListResponse.to_json(analytics_data_api.SheetExportAudienceListResponse())
        request = analytics_data_api.SheetExportAudienceListRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = analytics_data_api.SheetExportAudienceListResponse()
        client.sheet_export_audience_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_sheet_export_audience_list_rest_bad_request(transport: str='rest', request_type=analytics_data_api.SheetExportAudienceListRequest):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'properties/sample1/audienceLists/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.sheet_export_audience_list(request)

def test_sheet_export_audience_list_rest_flattened():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.SheetExportAudienceListResponse()
        sample_request = {'name': 'properties/sample1/audienceLists/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.SheetExportAudienceListResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.sheet_export_audience_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=properties/*/audienceLists/*}:exportSheet' % client.transport._host, args[1])

def test_sheet_export_audience_list_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.sheet_export_audience_list(analytics_data_api.SheetExportAudienceListRequest(), name='name_value')

def test_sheet_export_audience_list_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [analytics_data_api.GetAudienceListRequest, dict])
def test_get_audience_list_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'properties/sample1/audienceLists/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.AudienceList(name='name_value', audience='audience_value', audience_display_name='audience_display_name_value', state=analytics_data_api.AudienceList.State.CREATING, creation_quota_tokens_charged=3070, row_count=992, error_message='error_message_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.AudienceList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_audience_list(request)
    assert isinstance(response, analytics_data_api.AudienceList)
    assert response.name == 'name_value'
    assert response.audience == 'audience_value'
    assert response.audience_display_name == 'audience_display_name_value'
    assert response.state == analytics_data_api.AudienceList.State.CREATING
    assert response.creation_quota_tokens_charged == 3070
    assert response.row_count == 992
    assert response.error_message == 'error_message_value'

def test_get_audience_list_rest_required_fields(request_type=analytics_data_api.GetAudienceListRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlphaAnalyticsDataRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_audience_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = analytics_data_api.AudienceList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = analytics_data_api.AudienceList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_audience_list(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_audience_list_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_audience_list._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_audience_list_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlphaAnalyticsDataRestInterceptor())
    client = AlphaAnalyticsDataClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'post_get_audience_list') as post, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'pre_get_audience_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = analytics_data_api.GetAudienceListRequest.pb(analytics_data_api.GetAudienceListRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = analytics_data_api.AudienceList.to_json(analytics_data_api.AudienceList())
        request = analytics_data_api.GetAudienceListRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = analytics_data_api.AudienceList()
        client.get_audience_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_audience_list_rest_bad_request(transport: str='rest', request_type=analytics_data_api.GetAudienceListRequest):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'properties/sample1/audienceLists/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_audience_list(request)

def test_get_audience_list_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.AudienceList()
        sample_request = {'name': 'properties/sample1/audienceLists/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.AudienceList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_audience_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=properties/*/audienceLists/*}' % client.transport._host, args[1])

def test_get_audience_list_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_audience_list(analytics_data_api.GetAudienceListRequest(), name='name_value')

def test_get_audience_list_rest_error():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [analytics_data_api.ListAudienceListsRequest, dict])
def test_list_audience_lists_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'properties/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.ListAudienceListsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.ListAudienceListsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_audience_lists(request)
    assert isinstance(response, pagers.ListAudienceListsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_audience_lists_rest_required_fields(request_type=analytics_data_api.ListAudienceListsRequest):
    if False:
        return 10
    transport_class = transports.AlphaAnalyticsDataRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_audience_lists._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_audience_lists._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = analytics_data_api.ListAudienceListsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = analytics_data_api.ListAudienceListsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_audience_lists(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_audience_lists_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_audience_lists._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_audience_lists_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlphaAnalyticsDataRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlphaAnalyticsDataRestInterceptor())
    client = AlphaAnalyticsDataClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'post_list_audience_lists') as post, mock.patch.object(transports.AlphaAnalyticsDataRestInterceptor, 'pre_list_audience_lists') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = analytics_data_api.ListAudienceListsRequest.pb(analytics_data_api.ListAudienceListsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = analytics_data_api.ListAudienceListsResponse.to_json(analytics_data_api.ListAudienceListsResponse())
        request = analytics_data_api.ListAudienceListsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = analytics_data_api.ListAudienceListsResponse()
        client.list_audience_lists(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_audience_lists_rest_bad_request(transport: str='rest', request_type=analytics_data_api.ListAudienceListsRequest):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'properties/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_audience_lists(request)

def test_list_audience_lists_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = analytics_data_api.ListAudienceListsResponse()
        sample_request = {'parent': 'properties/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = analytics_data_api.ListAudienceListsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_audience_lists(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=properties/*}/audienceLists' % client.transport._host, args[1])

def test_list_audience_lists_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_audience_lists(analytics_data_api.ListAudienceListsRequest(), parent='parent_value')

def test_list_audience_lists_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList(), analytics_data_api.AudienceList()], next_page_token='abc'), analytics_data_api.ListAudienceListsResponse(audience_lists=[], next_page_token='def'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList()], next_page_token='ghi'), analytics_data_api.ListAudienceListsResponse(audience_lists=[analytics_data_api.AudienceList(), analytics_data_api.AudienceList()]))
        response = response + response
        response = tuple((analytics_data_api.ListAudienceListsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'properties/sample1'}
        pager = client.list_audience_lists(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, analytics_data_api.AudienceList) for i in results))
        pages = list(client.list_audience_lists(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.AlphaAnalyticsDataGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AlphaAnalyticsDataGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlphaAnalyticsDataClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AlphaAnalyticsDataGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AlphaAnalyticsDataClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AlphaAnalyticsDataClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AlphaAnalyticsDataGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlphaAnalyticsDataClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.AlphaAnalyticsDataGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AlphaAnalyticsDataClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.AlphaAnalyticsDataGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AlphaAnalyticsDataGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, transports.AlphaAnalyticsDataRestTransport])
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
        i = 10
        return i + 15
    transport = AlphaAnalyticsDataClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AlphaAnalyticsDataGrpcTransport)

def test_alpha_analytics_data_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AlphaAnalyticsDataTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_alpha_analytics_data_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.analytics.data_v1alpha.services.alpha_analytics_data.transports.AlphaAnalyticsDataTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AlphaAnalyticsDataTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('run_funnel_report', 'create_audience_list', 'query_audience_list', 'sheet_export_audience_list', 'get_audience_list', 'list_audience_lists')
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

def test_alpha_analytics_data_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.analytics.data_v1alpha.services.alpha_analytics_data.transports.AlphaAnalyticsDataTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AlphaAnalyticsDataTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/analytics', 'https://www.googleapis.com/auth/analytics.readonly', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/spreadsheets'), quota_project_id='octopus')

def test_alpha_analytics_data_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.analytics.data_v1alpha.services.alpha_analytics_data.transports.AlphaAnalyticsDataTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AlphaAnalyticsDataTransport()
        adc.assert_called_once()

def test_alpha_analytics_data_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AlphaAnalyticsDataClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/analytics', 'https://www.googleapis.com/auth/analytics.readonly', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/spreadsheets'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataGrpcAsyncIOTransport])
def test_alpha_analytics_data_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/analytics', 'https://www.googleapis.com/auth/analytics.readonly', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/spreadsheets'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataGrpcAsyncIOTransport, transports.AlphaAnalyticsDataRestTransport])
def test_alpha_analytics_data_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AlphaAnalyticsDataGrpcTransport, grpc_helpers), (transports.AlphaAnalyticsDataGrpcAsyncIOTransport, grpc_helpers_async)])
def test_alpha_analytics_data_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('analyticsdata.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/analytics', 'https://www.googleapis.com/auth/analytics.readonly', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/spreadsheets'), scopes=['1', '2'], default_host='analyticsdata.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataGrpcAsyncIOTransport])
def test_alpha_analytics_data_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_alpha_analytics_data_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AlphaAnalyticsDataRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_alpha_analytics_data_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_alpha_analytics_data_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='analyticsdata.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('analyticsdata.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://analyticsdata.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_alpha_analytics_data_host_with_port(transport_name):
    if False:
        return 10
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='analyticsdata.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('analyticsdata.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://analyticsdata.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_alpha_analytics_data_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AlphaAnalyticsDataClient(credentials=creds1, transport=transport_name)
    client2 = AlphaAnalyticsDataClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.run_funnel_report._session
    session2 = client2.transport.run_funnel_report._session
    assert session1 != session2
    session1 = client1.transport.create_audience_list._session
    session2 = client2.transport.create_audience_list._session
    assert session1 != session2
    session1 = client1.transport.query_audience_list._session
    session2 = client2.transport.query_audience_list._session
    assert session1 != session2
    session1 = client1.transport.sheet_export_audience_list._session
    session2 = client2.transport.sheet_export_audience_list._session
    assert session1 != session2
    session1 = client1.transport.get_audience_list._session
    session2 = client2.transport.get_audience_list._session
    assert session1 != session2
    session1 = client1.transport.list_audience_lists._session
    session2 = client2.transport.list_audience_lists._session
    assert session1 != session2

def test_alpha_analytics_data_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AlphaAnalyticsDataGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_alpha_analytics_data_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AlphaAnalyticsDataGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataGrpcAsyncIOTransport])
def test_alpha_analytics_data_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AlphaAnalyticsDataGrpcTransport, transports.AlphaAnalyticsDataGrpcAsyncIOTransport])
def test_alpha_analytics_data_transport_channel_mtls_with_adc(transport_class):
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

def test_alpha_analytics_data_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_alpha_analytics_data_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_audience_list_path():
    if False:
        i = 10
        return i + 15
    property = 'squid'
    audience_list = 'clam'
    expected = 'properties/{property}/audienceLists/{audience_list}'.format(property=property, audience_list=audience_list)
    actual = AlphaAnalyticsDataClient.audience_list_path(property, audience_list)
    assert expected == actual

def test_parse_audience_list_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'property': 'whelk', 'audience_list': 'octopus'}
    path = AlphaAnalyticsDataClient.audience_list_path(**expected)
    actual = AlphaAnalyticsDataClient.parse_audience_list_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AlphaAnalyticsDataClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'nudibranch'}
    path = AlphaAnalyticsDataClient.common_billing_account_path(**expected)
    actual = AlphaAnalyticsDataClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AlphaAnalyticsDataClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'mussel'}
    path = AlphaAnalyticsDataClient.common_folder_path(**expected)
    actual = AlphaAnalyticsDataClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AlphaAnalyticsDataClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = AlphaAnalyticsDataClient.common_organization_path(**expected)
    actual = AlphaAnalyticsDataClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = AlphaAnalyticsDataClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'abalone'}
    path = AlphaAnalyticsDataClient.common_project_path(**expected)
    actual = AlphaAnalyticsDataClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AlphaAnalyticsDataClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = AlphaAnalyticsDataClient.common_location_path(**expected)
    actual = AlphaAnalyticsDataClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AlphaAnalyticsDataTransport, '_prep_wrapped_messages') as prep:
        client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AlphaAnalyticsDataTransport, '_prep_wrapped_messages') as prep:
        transport_class = AlphaAnalyticsDataClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AlphaAnalyticsDataAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = AlphaAnalyticsDataClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AlphaAnalyticsDataClient, transports.AlphaAnalyticsDataGrpcTransport), (AlphaAnalyticsDataAsyncClient, transports.AlphaAnalyticsDataGrpcAsyncIOTransport)])
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