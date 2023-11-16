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
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.monitoring_v3.services.alert_policy_service import AlertPolicyServiceAsyncClient, AlertPolicyServiceClient, pagers, transports
from google.cloud.monitoring_v3.types import alert, alert_service, common, mutation_record

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert AlertPolicyServiceClient._get_default_mtls_endpoint(None) is None
    assert AlertPolicyServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AlertPolicyServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AlertPolicyServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AlertPolicyServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AlertPolicyServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AlertPolicyServiceClient, 'grpc'), (AlertPolicyServiceAsyncClient, 'grpc_asyncio')])
def test_alert_policy_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AlertPolicyServiceGrpcTransport, 'grpc'), (transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_alert_policy_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AlertPolicyServiceClient, 'grpc'), (AlertPolicyServiceAsyncClient, 'grpc_asyncio')])
def test_alert_policy_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'monitoring.googleapis.com:443'

def test_alert_policy_service_client_get_transport_class():
    if False:
        return 10
    transport = AlertPolicyServiceClient.get_transport_class()
    available_transports = [transports.AlertPolicyServiceGrpcTransport]
    assert transport in available_transports
    transport = AlertPolicyServiceClient.get_transport_class('grpc')
    assert transport == transports.AlertPolicyServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport, 'grpc'), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(AlertPolicyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlertPolicyServiceClient))
@mock.patch.object(AlertPolicyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlertPolicyServiceAsyncClient))
def test_alert_policy_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(AlertPolicyServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AlertPolicyServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport, 'grpc', 'true'), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport, 'grpc', 'false'), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(AlertPolicyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlertPolicyServiceClient))
@mock.patch.object(AlertPolicyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlertPolicyServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_alert_policy_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AlertPolicyServiceClient, AlertPolicyServiceAsyncClient])
@mock.patch.object(AlertPolicyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlertPolicyServiceClient))
@mock.patch.object(AlertPolicyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlertPolicyServiceAsyncClient))
def test_alert_policy_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport, 'grpc'), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_alert_policy_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport, 'grpc', grpc_helpers), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_alert_policy_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_alert_policy_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.monitoring_v3.services.alert_policy_service.transports.AlertPolicyServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AlertPolicyServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport, 'grpc', grpc_helpers), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_alert_policy_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=None, default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [alert_service.ListAlertPoliciesRequest, dict])
def test_list_alert_policies(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.return_value = alert_service.ListAlertPoliciesResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_alert_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.ListAlertPoliciesRequest()
    assert isinstance(response, pagers.ListAlertPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_alert_policies_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        client.list_alert_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.ListAlertPoliciesRequest()

@pytest.mark.asyncio
async def test_list_alert_policies_async(transport: str='grpc_asyncio', request_type=alert_service.ListAlertPoliciesRequest):
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert_service.ListAlertPoliciesResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_alert_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.ListAlertPoliciesRequest()
    assert isinstance(response, pagers.ListAlertPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_alert_policies_async_from_dict():
    await test_list_alert_policies_async(request_type=dict)

def test_list_alert_policies_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.ListAlertPoliciesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.return_value = alert_service.ListAlertPoliciesResponse()
        client.list_alert_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_alert_policies_field_headers_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.ListAlertPoliciesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert_service.ListAlertPoliciesResponse())
        await client.list_alert_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_list_alert_policies_flattened():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.return_value = alert_service.ListAlertPoliciesResponse()
        client.list_alert_policies(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_list_alert_policies_flattened_error():
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_alert_policies(alert_service.ListAlertPoliciesRequest(), name='name_value')

@pytest.mark.asyncio
async def test_list_alert_policies_flattened_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.return_value = alert_service.ListAlertPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert_service.ListAlertPoliciesResponse())
        response = await client.list_alert_policies(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_alert_policies_flattened_error_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_alert_policies(alert_service.ListAlertPoliciesRequest(), name='name_value')

def test_list_alert_policies_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.side_effect = (alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy(), alert.AlertPolicy()], next_page_token='abc'), alert_service.ListAlertPoliciesResponse(alert_policies=[], next_page_token='def'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy()], next_page_token='ghi'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', ''),)),)
        pager = client.list_alert_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, alert.AlertPolicy) for i in results))

def test_list_alert_policies_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__') as call:
        call.side_effect = (alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy(), alert.AlertPolicy()], next_page_token='abc'), alert_service.ListAlertPoliciesResponse(alert_policies=[], next_page_token='def'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy()], next_page_token='ghi'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy()]), RuntimeError)
        pages = list(client.list_alert_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_alert_policies_async_pager():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy(), alert.AlertPolicy()], next_page_token='abc'), alert_service.ListAlertPoliciesResponse(alert_policies=[], next_page_token='def'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy()], next_page_token='ghi'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy()]), RuntimeError)
        async_pager = await client.list_alert_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, alert.AlertPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_alert_policies_async_pages():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_alert_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy(), alert.AlertPolicy()], next_page_token='abc'), alert_service.ListAlertPoliciesResponse(alert_policies=[], next_page_token='def'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy()], next_page_token='ghi'), alert_service.ListAlertPoliciesResponse(alert_policies=[alert.AlertPolicy(), alert.AlertPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_alert_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [alert_service.GetAlertPolicyRequest, dict])
def test_get_alert_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy(name='name_value', display_name='display_name_value', combiner=alert.AlertPolicy.ConditionCombinerType.AND, notification_channels=['notification_channels_value'])
        response = client.get_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.GetAlertPolicyRequest()
    assert isinstance(response, alert.AlertPolicy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.combiner == alert.AlertPolicy.ConditionCombinerType.AND
    assert response.notification_channels == ['notification_channels_value']

def test_get_alert_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        client.get_alert_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.GetAlertPolicyRequest()

@pytest.mark.asyncio
async def test_get_alert_policy_async(transport: str='grpc_asyncio', request_type=alert_service.GetAlertPolicyRequest):
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy(name='name_value', display_name='display_name_value', combiner=alert.AlertPolicy.ConditionCombinerType.AND, notification_channels=['notification_channels_value']))
        response = await client.get_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.GetAlertPolicyRequest()
    assert isinstance(response, alert.AlertPolicy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.combiner == alert.AlertPolicy.ConditionCombinerType.AND
    assert response.notification_channels == ['notification_channels_value']

@pytest.mark.asyncio
async def test_get_alert_policy_async_from_dict():
    await test_get_alert_policy_async(request_type=dict)

def test_get_alert_policy_field_headers():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.GetAlertPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        client.get_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_alert_policy_field_headers_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.GetAlertPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy())
        await client.get_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_alert_policy_flattened():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        client.get_alert_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_alert_policy_flattened_error():
    if False:
        return 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_alert_policy(alert_service.GetAlertPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_alert_policy_flattened_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy())
        response = await client.get_alert_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_alert_policy_flattened_error_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_alert_policy(alert_service.GetAlertPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [alert_service.CreateAlertPolicyRequest, dict])
def test_create_alert_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy(name='name_value', display_name='display_name_value', combiner=alert.AlertPolicy.ConditionCombinerType.AND, notification_channels=['notification_channels_value'])
        response = client.create_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.CreateAlertPolicyRequest()
    assert isinstance(response, alert.AlertPolicy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.combiner == alert.AlertPolicy.ConditionCombinerType.AND
    assert response.notification_channels == ['notification_channels_value']

def test_create_alert_policy_empty_call():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        client.create_alert_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.CreateAlertPolicyRequest()

@pytest.mark.asyncio
async def test_create_alert_policy_async(transport: str='grpc_asyncio', request_type=alert_service.CreateAlertPolicyRequest):
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy(name='name_value', display_name='display_name_value', combiner=alert.AlertPolicy.ConditionCombinerType.AND, notification_channels=['notification_channels_value']))
        response = await client.create_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.CreateAlertPolicyRequest()
    assert isinstance(response, alert.AlertPolicy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.combiner == alert.AlertPolicy.ConditionCombinerType.AND
    assert response.notification_channels == ['notification_channels_value']

@pytest.mark.asyncio
async def test_create_alert_policy_async_from_dict():
    await test_create_alert_policy_async(request_type=dict)

def test_create_alert_policy_field_headers():
    if False:
        return 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.CreateAlertPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        client.create_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_alert_policy_field_headers_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.CreateAlertPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy())
        await client.create_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_create_alert_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        client.create_alert_policy(name='name_value', alert_policy=alert.AlertPolicy(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].alert_policy
        mock_val = alert.AlertPolicy(name='name_value')
        assert arg == mock_val

def test_create_alert_policy_flattened_error():
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_alert_policy(alert_service.CreateAlertPolicyRequest(), name='name_value', alert_policy=alert.AlertPolicy(name='name_value'))

@pytest.mark.asyncio
async def test_create_alert_policy_flattened_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy())
        response = await client.create_alert_policy(name='name_value', alert_policy=alert.AlertPolicy(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].alert_policy
        mock_val = alert.AlertPolicy(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_alert_policy_flattened_error_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_alert_policy(alert_service.CreateAlertPolicyRequest(), name='name_value', alert_policy=alert.AlertPolicy(name='name_value'))

@pytest.mark.parametrize('request_type', [alert_service.DeleteAlertPolicyRequest, dict])
def test_delete_alert_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        call.return_value = None
        response = client.delete_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.DeleteAlertPolicyRequest()
    assert response is None

def test_delete_alert_policy_empty_call():
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        client.delete_alert_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.DeleteAlertPolicyRequest()

@pytest.mark.asyncio
async def test_delete_alert_policy_async(transport: str='grpc_asyncio', request_type=alert_service.DeleteAlertPolicyRequest):
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.DeleteAlertPolicyRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_alert_policy_async_from_dict():
    await test_delete_alert_policy_async(request_type=dict)

def test_delete_alert_policy_field_headers():
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.DeleteAlertPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        call.return_value = None
        client.delete_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_alert_policy_field_headers_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.DeleteAlertPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_alert_policy_flattened():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        call.return_value = None
        client.delete_alert_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_alert_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_alert_policy(alert_service.DeleteAlertPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_alert_policy_flattened_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_alert_policy), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_alert_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_alert_policy_flattened_error_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_alert_policy(alert_service.DeleteAlertPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [alert_service.UpdateAlertPolicyRequest, dict])
def test_update_alert_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy(name='name_value', display_name='display_name_value', combiner=alert.AlertPolicy.ConditionCombinerType.AND, notification_channels=['notification_channels_value'])
        response = client.update_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.UpdateAlertPolicyRequest()
    assert isinstance(response, alert.AlertPolicy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.combiner == alert.AlertPolicy.ConditionCombinerType.AND
    assert response.notification_channels == ['notification_channels_value']

def test_update_alert_policy_empty_call():
    if False:
        while True:
            i = 10
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        client.update_alert_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.UpdateAlertPolicyRequest()

@pytest.mark.asyncio
async def test_update_alert_policy_async(transport: str='grpc_asyncio', request_type=alert_service.UpdateAlertPolicyRequest):
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy(name='name_value', display_name='display_name_value', combiner=alert.AlertPolicy.ConditionCombinerType.AND, notification_channels=['notification_channels_value']))
        response = await client.update_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == alert_service.UpdateAlertPolicyRequest()
    assert isinstance(response, alert.AlertPolicy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.combiner == alert.AlertPolicy.ConditionCombinerType.AND
    assert response.notification_channels == ['notification_channels_value']

@pytest.mark.asyncio
async def test_update_alert_policy_async_from_dict():
    await test_update_alert_policy_async(request_type=dict)

def test_update_alert_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.UpdateAlertPolicyRequest()
    request.alert_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        client.update_alert_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'alert_policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_alert_policy_field_headers_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = alert_service.UpdateAlertPolicyRequest()
    request.alert_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy())
        await client.update_alert_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'alert_policy.name=name_value') in kw['metadata']

def test_update_alert_policy_flattened():
    if False:
        i = 10
        return i + 15
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        client.update_alert_policy(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), alert_policy=alert.AlertPolicy(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].alert_policy
        mock_val = alert.AlertPolicy(name='name_value')
        assert arg == mock_val

def test_update_alert_policy_flattened_error():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_alert_policy(alert_service.UpdateAlertPolicyRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), alert_policy=alert.AlertPolicy(name='name_value'))

@pytest.mark.asyncio
async def test_update_alert_policy_flattened_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_alert_policy), '__call__') as call:
        call.return_value = alert.AlertPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(alert.AlertPolicy())
        response = await client.update_alert_policy(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), alert_policy=alert.AlertPolicy(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].alert_policy
        mock_val = alert.AlertPolicy(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_alert_policy_flattened_error_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_alert_policy(alert_service.UpdateAlertPolicyRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), alert_policy=alert.AlertPolicy(name='name_value'))

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.AlertPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AlertPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlertPolicyServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AlertPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AlertPolicyServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AlertPolicyServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AlertPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlertPolicyServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.AlertPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AlertPolicyServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.AlertPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AlertPolicyServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AlertPolicyServiceGrpcTransport, transports.AlertPolicyServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = AlertPolicyServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AlertPolicyServiceGrpcTransport)

def test_alert_policy_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AlertPolicyServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_alert_policy_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.monitoring_v3.services.alert_policy_service.transports.AlertPolicyServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AlertPolicyServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_alert_policies', 'get_alert_policy', 'create_alert_policy', 'delete_alert_policy', 'update_alert_policy')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_alert_policy_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.monitoring_v3.services.alert_policy_service.transports.AlertPolicyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AlertPolicyServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

def test_alert_policy_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.monitoring_v3.services.alert_policy_service.transports.AlertPolicyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AlertPolicyServiceTransport()
        adc.assert_called_once()

def test_alert_policy_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AlertPolicyServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AlertPolicyServiceGrpcTransport, transports.AlertPolicyServiceGrpcAsyncIOTransport])
def test_alert_policy_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AlertPolicyServiceGrpcTransport, transports.AlertPolicyServiceGrpcAsyncIOTransport])
def test_alert_policy_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AlertPolicyServiceGrpcTransport, grpc_helpers), (transports.AlertPolicyServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_alert_policy_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=['1', '2'], default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AlertPolicyServiceGrpcTransport, transports.AlertPolicyServiceGrpcAsyncIOTransport])
def test_alert_policy_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_alert_policy_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_alert_policy_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:8000'

def test_alert_policy_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AlertPolicyServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_alert_policy_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AlertPolicyServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AlertPolicyServiceGrpcTransport, transports.AlertPolicyServiceGrpcAsyncIOTransport])
def test_alert_policy_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AlertPolicyServiceGrpcTransport, transports.AlertPolicyServiceGrpcAsyncIOTransport])
def test_alert_policy_service_transport_channel_mtls_with_adc(transport_class):
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

def test_alert_policy_path():
    if False:
        print('Hello World!')
    project = 'squid'
    alert_policy = 'clam'
    expected = 'projects/{project}/alertPolicies/{alert_policy}'.format(project=project, alert_policy=alert_policy)
    actual = AlertPolicyServiceClient.alert_policy_path(project, alert_policy)
    assert expected == actual

def test_parse_alert_policy_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'alert_policy': 'octopus'}
    path = AlertPolicyServiceClient.alert_policy_path(**expected)
    actual = AlertPolicyServiceClient.parse_alert_policy_path(path)
    assert expected == actual

def test_alert_policy_condition_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    alert_policy = 'nudibranch'
    condition = 'cuttlefish'
    expected = 'projects/{project}/alertPolicies/{alert_policy}/conditions/{condition}'.format(project=project, alert_policy=alert_policy, condition=condition)
    actual = AlertPolicyServiceClient.alert_policy_condition_path(project, alert_policy, condition)
    assert expected == actual

def test_parse_alert_policy_condition_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'alert_policy': 'winkle', 'condition': 'nautilus'}
    path = AlertPolicyServiceClient.alert_policy_condition_path(**expected)
    actual = AlertPolicyServiceClient.parse_alert_policy_condition_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AlertPolicyServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'abalone'}
    path = AlertPolicyServiceClient.common_billing_account_path(**expected)
    actual = AlertPolicyServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AlertPolicyServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = AlertPolicyServiceClient.common_folder_path(**expected)
    actual = AlertPolicyServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AlertPolicyServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'octopus'}
    path = AlertPolicyServiceClient.common_organization_path(**expected)
    actual = AlertPolicyServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = AlertPolicyServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch'}
    path = AlertPolicyServiceClient.common_project_path(**expected)
    actual = AlertPolicyServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AlertPolicyServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = AlertPolicyServiceClient.common_location_path(**expected)
    actual = AlertPolicyServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AlertPolicyServiceTransport, '_prep_wrapped_messages') as prep:
        client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AlertPolicyServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = AlertPolicyServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AlertPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        i = 10
        return i + 15
    transports = ['grpc']
    for transport in transports:
        client = AlertPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AlertPolicyServiceClient, transports.AlertPolicyServiceGrpcTransport), (AlertPolicyServiceAsyncClient, transports.AlertPolicyServiceGrpcAsyncIOTransport)])
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