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
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.webrisk_v1.services.web_risk_service import WebRiskServiceAsyncClient, WebRiskServiceClient, transports
from google.cloud.webrisk_v1.types import webrisk

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
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
    assert WebRiskServiceClient._get_default_mtls_endpoint(None) is None
    assert WebRiskServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert WebRiskServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert WebRiskServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert WebRiskServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert WebRiskServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(WebRiskServiceClient, 'grpc'), (WebRiskServiceAsyncClient, 'grpc_asyncio'), (WebRiskServiceClient, 'rest')])
def test_web_risk_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('webrisk.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.WebRiskServiceGrpcTransport, 'grpc'), (transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.WebRiskServiceRestTransport, 'rest')])
def test_web_risk_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(WebRiskServiceClient, 'grpc'), (WebRiskServiceAsyncClient, 'grpc_asyncio'), (WebRiskServiceClient, 'rest')])
def test_web_risk_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('webrisk.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com')

def test_web_risk_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = WebRiskServiceClient.get_transport_class()
    available_transports = [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceRestTransport]
    assert transport in available_transports
    transport = WebRiskServiceClient.get_transport_class('grpc')
    assert transport == transports.WebRiskServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WebRiskServiceClient, transports.WebRiskServiceGrpcTransport, 'grpc'), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (WebRiskServiceClient, transports.WebRiskServiceRestTransport, 'rest')])
@mock.patch.object(WebRiskServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceClient))
@mock.patch.object(WebRiskServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceAsyncClient))
def test_web_risk_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(WebRiskServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(WebRiskServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(WebRiskServiceClient, transports.WebRiskServiceGrpcTransport, 'grpc', 'true'), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (WebRiskServiceClient, transports.WebRiskServiceGrpcTransport, 'grpc', 'false'), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (WebRiskServiceClient, transports.WebRiskServiceRestTransport, 'rest', 'true'), (WebRiskServiceClient, transports.WebRiskServiceRestTransport, 'rest', 'false')])
@mock.patch.object(WebRiskServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceClient))
@mock.patch.object(WebRiskServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_web_risk_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [WebRiskServiceClient, WebRiskServiceAsyncClient])
@mock.patch.object(WebRiskServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceClient))
@mock.patch.object(WebRiskServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceAsyncClient))
def test_web_risk_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WebRiskServiceClient, transports.WebRiskServiceGrpcTransport, 'grpc'), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (WebRiskServiceClient, transports.WebRiskServiceRestTransport, 'rest')])
def test_web_risk_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WebRiskServiceClient, transports.WebRiskServiceGrpcTransport, 'grpc', grpc_helpers), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (WebRiskServiceClient, transports.WebRiskServiceRestTransport, 'rest', None)])
def test_web_risk_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_web_risk_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.webrisk_v1.services.web_risk_service.transports.WebRiskServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = WebRiskServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WebRiskServiceClient, transports.WebRiskServiceGrpcTransport, 'grpc', grpc_helpers), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_web_risk_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('webrisk.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='webrisk.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [webrisk.ComputeThreatListDiffRequest, dict])
def test_compute_threat_list_diff(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.compute_threat_list_diff), '__call__') as call:
        call.return_value = webrisk.ComputeThreatListDiffResponse(response_type=webrisk.ComputeThreatListDiffResponse.ResponseType.DIFF, new_version_token=b'new_version_token_blob')
        response = client.compute_threat_list_diff(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.ComputeThreatListDiffRequest()
    assert isinstance(response, webrisk.ComputeThreatListDiffResponse)
    assert response.response_type == webrisk.ComputeThreatListDiffResponse.ResponseType.DIFF
    assert response.new_version_token == b'new_version_token_blob'

def test_compute_threat_list_diff_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.compute_threat_list_diff), '__call__') as call:
        client.compute_threat_list_diff()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.ComputeThreatListDiffRequest()

@pytest.mark.asyncio
async def test_compute_threat_list_diff_async(transport: str='grpc_asyncio', request_type=webrisk.ComputeThreatListDiffRequest):
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.compute_threat_list_diff), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.ComputeThreatListDiffResponse(response_type=webrisk.ComputeThreatListDiffResponse.ResponseType.DIFF, new_version_token=b'new_version_token_blob'))
        response = await client.compute_threat_list_diff(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.ComputeThreatListDiffRequest()
    assert isinstance(response, webrisk.ComputeThreatListDiffResponse)
    assert response.response_type == webrisk.ComputeThreatListDiffResponse.ResponseType.DIFF
    assert response.new_version_token == b'new_version_token_blob'

@pytest.mark.asyncio
async def test_compute_threat_list_diff_async_from_dict():
    await test_compute_threat_list_diff_async(request_type=dict)

def test_compute_threat_list_diff_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.compute_threat_list_diff), '__call__') as call:
        call.return_value = webrisk.ComputeThreatListDiffResponse()
        client.compute_threat_list_diff(threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].threat_type
        mock_val = webrisk.ThreatType.MALWARE
        assert arg == mock_val
        arg = args[0].version_token
        mock_val = b'version_token_blob'
        assert arg == mock_val
        arg = args[0].constraints
        mock_val = webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687)
        assert arg == mock_val

def test_compute_threat_list_diff_flattened_error():
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.compute_threat_list_diff(webrisk.ComputeThreatListDiffRequest(), threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))

@pytest.mark.asyncio
async def test_compute_threat_list_diff_flattened_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.compute_threat_list_diff), '__call__') as call:
        call.return_value = webrisk.ComputeThreatListDiffResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.ComputeThreatListDiffResponse())
        response = await client.compute_threat_list_diff(threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].threat_type
        mock_val = webrisk.ThreatType.MALWARE
        assert arg == mock_val
        arg = args[0].version_token
        mock_val = b'version_token_blob'
        assert arg == mock_val
        arg = args[0].constraints
        mock_val = webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687)
        assert arg == mock_val

@pytest.mark.asyncio
async def test_compute_threat_list_diff_flattened_error_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.compute_threat_list_diff(webrisk.ComputeThreatListDiffRequest(), threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))

@pytest.mark.parametrize('request_type', [webrisk.SearchUrisRequest, dict])
def test_search_uris(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_uris), '__call__') as call:
        call.return_value = webrisk.SearchUrisResponse()
        response = client.search_uris(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchUrisRequest()
    assert isinstance(response, webrisk.SearchUrisResponse)

def test_search_uris_empty_call():
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_uris), '__call__') as call:
        client.search_uris()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchUrisRequest()

@pytest.mark.asyncio
async def test_search_uris_async(transport: str='grpc_asyncio', request_type=webrisk.SearchUrisRequest):
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_uris), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.SearchUrisResponse())
        response = await client.search_uris(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchUrisRequest()
    assert isinstance(response, webrisk.SearchUrisResponse)

@pytest.mark.asyncio
async def test_search_uris_async_from_dict():
    await test_search_uris_async(request_type=dict)

def test_search_uris_flattened():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_uris), '__call__') as call:
        call.return_value = webrisk.SearchUrisResponse()
        client.search_uris(uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].uri
        mock_val = 'uri_value'
        assert arg == mock_val
        arg = args[0].threat_types
        mock_val = [webrisk.ThreatType.MALWARE]
        assert arg == mock_val

def test_search_uris_flattened_error():
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_uris(webrisk.SearchUrisRequest(), uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.asyncio
async def test_search_uris_flattened_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_uris), '__call__') as call:
        call.return_value = webrisk.SearchUrisResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.SearchUrisResponse())
        response = await client.search_uris(uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].uri
        mock_val = 'uri_value'
        assert arg == mock_val
        arg = args[0].threat_types
        mock_val = [webrisk.ThreatType.MALWARE]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_uris_flattened_error_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_uris(webrisk.SearchUrisRequest(), uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.parametrize('request_type', [webrisk.SearchHashesRequest, dict])
def test_search_hashes(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_hashes), '__call__') as call:
        call.return_value = webrisk.SearchHashesResponse()
        response = client.search_hashes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchHashesRequest()
    assert isinstance(response, webrisk.SearchHashesResponse)

def test_search_hashes_empty_call():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_hashes), '__call__') as call:
        client.search_hashes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchHashesRequest()

@pytest.mark.asyncio
async def test_search_hashes_async(transport: str='grpc_asyncio', request_type=webrisk.SearchHashesRequest):
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_hashes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.SearchHashesResponse())
        response = await client.search_hashes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchHashesRequest()
    assert isinstance(response, webrisk.SearchHashesResponse)

@pytest.mark.asyncio
async def test_search_hashes_async_from_dict():
    await test_search_hashes_async(request_type=dict)

def test_search_hashes_flattened():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_hashes), '__call__') as call:
        call.return_value = webrisk.SearchHashesResponse()
        client.search_hashes(hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].hash_prefix
        mock_val = b'hash_prefix_blob'
        assert arg == mock_val
        arg = args[0].threat_types
        mock_val = [webrisk.ThreatType.MALWARE]
        assert arg == mock_val

def test_search_hashes_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_hashes(webrisk.SearchHashesRequest(), hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.asyncio
async def test_search_hashes_flattened_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_hashes), '__call__') as call:
        call.return_value = webrisk.SearchHashesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.SearchHashesResponse())
        response = await client.search_hashes(hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].hash_prefix
        mock_val = b'hash_prefix_blob'
        assert arg == mock_val
        arg = args[0].threat_types
        mock_val = [webrisk.ThreatType.MALWARE]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_hashes_flattened_error_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_hashes(webrisk.SearchHashesRequest(), hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.parametrize('request_type', [webrisk.CreateSubmissionRequest, dict])
def test_create_submission(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        call.return_value = webrisk.Submission(uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])
        response = client.create_submission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.CreateSubmissionRequest()
    assert isinstance(response, webrisk.Submission)
    assert response.uri == 'uri_value'
    assert response.threat_types == [webrisk.ThreatType.MALWARE]

def test_create_submission_empty_call():
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        client.create_submission()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.CreateSubmissionRequest()

@pytest.mark.asyncio
async def test_create_submission_async(transport: str='grpc_asyncio', request_type=webrisk.CreateSubmissionRequest):
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.Submission(uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE]))
        response = await client.create_submission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.CreateSubmissionRequest()
    assert isinstance(response, webrisk.Submission)
    assert response.uri == 'uri_value'
    assert response.threat_types == [webrisk.ThreatType.MALWARE]

@pytest.mark.asyncio
async def test_create_submission_async_from_dict():
    await test_create_submission_async(request_type=dict)

def test_create_submission_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = webrisk.CreateSubmissionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        call.return_value = webrisk.Submission()
        client.create_submission(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_submission_field_headers_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = webrisk.CreateSubmissionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.Submission())
        await client.create_submission(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_submission_flattened():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        call.return_value = webrisk.Submission()
        client.create_submission(parent='parent_value', submission=webrisk.Submission(uri='uri_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].submission
        mock_val = webrisk.Submission(uri='uri_value')
        assert arg == mock_val

def test_create_submission_flattened_error():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_submission(webrisk.CreateSubmissionRequest(), parent='parent_value', submission=webrisk.Submission(uri='uri_value'))

@pytest.mark.asyncio
async def test_create_submission_flattened_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_submission), '__call__') as call:
        call.return_value = webrisk.Submission()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(webrisk.Submission())
        response = await client.create_submission(parent='parent_value', submission=webrisk.Submission(uri='uri_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].submission
        mock_val = webrisk.Submission(uri='uri_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_submission_flattened_error_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_submission(webrisk.CreateSubmissionRequest(), parent='parent_value', submission=webrisk.Submission(uri='uri_value'))

@pytest.mark.parametrize('request_type', [webrisk.SubmitUriRequest, dict])
def test_submit_uri(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_uri), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.submit_uri(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SubmitUriRequest()
    assert isinstance(response, future.Future)

def test_submit_uri_empty_call():
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.submit_uri), '__call__') as call:
        client.submit_uri()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SubmitUriRequest()

@pytest.mark.asyncio
async def test_submit_uri_async(transport: str='grpc_asyncio', request_type=webrisk.SubmitUriRequest):
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_uri), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.submit_uri(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SubmitUriRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_submit_uri_async_from_dict():
    await test_submit_uri_async(request_type=dict)

def test_submit_uri_field_headers():
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = webrisk.SubmitUriRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.submit_uri), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.submit_uri(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_submit_uri_field_headers_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = webrisk.SubmitUriRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.submit_uri), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.submit_uri(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [webrisk.ComputeThreatListDiffRequest, dict])
def test_compute_threat_list_diff_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.ComputeThreatListDiffResponse(response_type=webrisk.ComputeThreatListDiffResponse.ResponseType.DIFF, new_version_token=b'new_version_token_blob')
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.ComputeThreatListDiffResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.compute_threat_list_diff(request)
    assert isinstance(response, webrisk.ComputeThreatListDiffResponse)
    assert response.response_type == webrisk.ComputeThreatListDiffResponse.ResponseType.DIFF
    assert response.new_version_token == b'new_version_token_blob'

def test_compute_threat_list_diff_rest_required_fields(request_type=webrisk.ComputeThreatListDiffRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WebRiskServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).compute_threat_list_diff._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).compute_threat_list_diff._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('constraints', 'threat_type', 'version_token'))
    jsonified_request.update(unset_fields)
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = webrisk.ComputeThreatListDiffResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = webrisk.ComputeThreatListDiffResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.compute_threat_list_diff(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_compute_threat_list_diff_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.compute_threat_list_diff._get_unset_required_fields({})
    assert set(unset_fields) == set(('constraints', 'threatType', 'versionToken')) & set(('threatType', 'constraints'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_compute_threat_list_diff_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceRestInterceptor())
    client = WebRiskServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'post_compute_threat_list_diff') as post, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'pre_compute_threat_list_diff') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = webrisk.ComputeThreatListDiffRequest.pb(webrisk.ComputeThreatListDiffRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = webrisk.ComputeThreatListDiffResponse.to_json(webrisk.ComputeThreatListDiffResponse())
        request = webrisk.ComputeThreatListDiffRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = webrisk.ComputeThreatListDiffResponse()
        client.compute_threat_list_diff(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_compute_threat_list_diff_rest_bad_request(transport: str='rest', request_type=webrisk.ComputeThreatListDiffRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.compute_threat_list_diff(request)

def test_compute_threat_list_diff_rest_flattened():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.ComputeThreatListDiffResponse()
        sample_request = {}
        mock_args = dict(threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.ComputeThreatListDiffResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.compute_threat_list_diff(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/threatLists:computeDiff' % client.transport._host, args[1])

def test_compute_threat_list_diff_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.compute_threat_list_diff(webrisk.ComputeThreatListDiffRequest(), threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))

def test_compute_threat_list_diff_rest_error():
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [webrisk.SearchUrisRequest, dict])
def test_search_uris_rest(request_type):
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.SearchUrisResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.SearchUrisResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_uris(request)
    assert isinstance(response, webrisk.SearchUrisResponse)

def test_search_uris_rest_required_fields(request_type=webrisk.SearchUrisRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WebRiskServiceRestTransport
    request_init = {}
    request_init['uri'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'uri' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_uris._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'uri' in jsonified_request
    assert jsonified_request['uri'] == request_init['uri']
    jsonified_request['uri'] = 'uri_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_uris._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('threat_types', 'uri'))
    jsonified_request.update(unset_fields)
    assert 'uri' in jsonified_request
    assert jsonified_request['uri'] == 'uri_value'
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = webrisk.SearchUrisResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = webrisk.SearchUrisResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_uris(request)
            expected_params = [('uri', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_uris_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_uris._get_unset_required_fields({})
    assert set(unset_fields) == set(('threatTypes', 'uri')) & set(('uri', 'threatTypes'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_uris_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceRestInterceptor())
    client = WebRiskServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'post_search_uris') as post, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'pre_search_uris') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = webrisk.SearchUrisRequest.pb(webrisk.SearchUrisRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = webrisk.SearchUrisResponse.to_json(webrisk.SearchUrisResponse())
        request = webrisk.SearchUrisRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = webrisk.SearchUrisResponse()
        client.search_uris(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_uris_rest_bad_request(transport: str='rest', request_type=webrisk.SearchUrisRequest):
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_uris(request)

def test_search_uris_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.SearchUrisResponse()
        sample_request = {}
        mock_args = dict(uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.SearchUrisResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_uris(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/uris:search' % client.transport._host, args[1])

def test_search_uris_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_uris(webrisk.SearchUrisRequest(), uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])

def test_search_uris_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [webrisk.SearchHashesRequest, dict])
def test_search_hashes_rest(request_type):
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.SearchHashesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.SearchHashesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_hashes(request)
    assert isinstance(response, webrisk.SearchHashesResponse)

def test_search_hashes_rest_required_fields(request_type=webrisk.SearchHashesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WebRiskServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_hashes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_hashes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('hash_prefix', 'threat_types'))
    jsonified_request.update(unset_fields)
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = webrisk.SearchHashesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = webrisk.SearchHashesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_hashes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_hashes_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_hashes._get_unset_required_fields({})
    assert set(unset_fields) == set(('hashPrefix', 'threatTypes')) & set(('threatTypes',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_hashes_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceRestInterceptor())
    client = WebRiskServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'post_search_hashes') as post, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'pre_search_hashes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = webrisk.SearchHashesRequest.pb(webrisk.SearchHashesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = webrisk.SearchHashesResponse.to_json(webrisk.SearchHashesResponse())
        request = webrisk.SearchHashesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = webrisk.SearchHashesResponse()
        client.search_hashes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_hashes_rest_bad_request(transport: str='rest', request_type=webrisk.SearchHashesRequest):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_hashes(request)

def test_search_hashes_rest_flattened():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.SearchHashesResponse()
        sample_request = {}
        mock_args = dict(hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.SearchHashesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_hashes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/hashes:search' % client.transport._host, args[1])

def test_search_hashes_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_hashes(webrisk.SearchHashesRequest(), hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])

def test_search_hashes_rest_error():
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [webrisk.CreateSubmissionRequest, dict])
def test_create_submission_rest(request_type):
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['submission'] = {'uri': 'uri_value', 'threat_types': [1]}
    test_field = webrisk.CreateSubmissionRequest.meta.fields['submission']

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
    for (field, value) in request_init['submission'].items():
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
                for i in range(0, len(request_init['submission'][field])):
                    del request_init['submission'][field][i][subfield]
            else:
                del request_init['submission'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.Submission(uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.Submission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_submission(request)
    assert isinstance(response, webrisk.Submission)
    assert response.uri == 'uri_value'
    assert response.threat_types == [webrisk.ThreatType.MALWARE]

def test_create_submission_rest_required_fields(request_type=webrisk.CreateSubmissionRequest):
    if False:
        return 10
    transport_class = transports.WebRiskServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_submission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_submission._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = webrisk.Submission()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = webrisk.Submission.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_submission(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_submission_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_submission._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'submission'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_submission_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceRestInterceptor())
    client = WebRiskServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'post_create_submission') as post, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'pre_create_submission') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = webrisk.CreateSubmissionRequest.pb(webrisk.CreateSubmissionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = webrisk.Submission.to_json(webrisk.Submission())
        request = webrisk.CreateSubmissionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = webrisk.Submission()
        client.create_submission(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_submission_rest_bad_request(transport: str='rest', request_type=webrisk.CreateSubmissionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_submission(request)

def test_create_submission_rest_flattened():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = webrisk.Submission()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', submission=webrisk.Submission(uri='uri_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = webrisk.Submission.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_submission(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/submissions' % client.transport._host, args[1])

def test_create_submission_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_submission(webrisk.CreateSubmissionRequest(), parent='parent_value', submission=webrisk.Submission(uri='uri_value'))

def test_create_submission_rest_error():
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [webrisk.SubmitUriRequest, dict])
def test_submit_uri_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.submit_uri(request)
    assert response.operation.name == 'operations/spam'

def test_submit_uri_rest_required_fields(request_type=webrisk.SubmitUriRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.WebRiskServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_uri._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_uri._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.submit_uri(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_submit_uri_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.submit_uri._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'submission'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_submit_uri_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.WebRiskServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceRestInterceptor())
    client = WebRiskServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WebRiskServiceRestInterceptor, 'post_submit_uri') as post, mock.patch.object(transports.WebRiskServiceRestInterceptor, 'pre_submit_uri') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = webrisk.SubmitUriRequest.pb(webrisk.SubmitUriRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = webrisk.SubmitUriRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.submit_uri(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_submit_uri_rest_bad_request(transport: str='rest', request_type=webrisk.SubmitUriRequest):
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.submit_uri(request)

def test_submit_uri_rest_error():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.WebRiskServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.WebRiskServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WebRiskServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.WebRiskServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WebRiskServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WebRiskServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.WebRiskServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WebRiskServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.WebRiskServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = WebRiskServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.WebRiskServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.WebRiskServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceGrpcAsyncIOTransport, transports.WebRiskServiceRestTransport])
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
    transport = WebRiskServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.WebRiskServiceGrpcTransport)

def test_web_risk_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.WebRiskServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_web_risk_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.webrisk_v1.services.web_risk_service.transports.WebRiskServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.WebRiskServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('compute_threat_list_diff', 'search_uris', 'search_hashes', 'create_submission', 'submit_uri', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_web_risk_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.webrisk_v1.services.web_risk_service.transports.WebRiskServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WebRiskServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_web_risk_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.webrisk_v1.services.web_risk_service.transports.WebRiskServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WebRiskServiceTransport()
        adc.assert_called_once()

def test_web_risk_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        WebRiskServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceGrpcAsyncIOTransport])
def test_web_risk_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceGrpcAsyncIOTransport, transports.WebRiskServiceRestTransport])
def test_web_risk_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.WebRiskServiceGrpcTransport, grpc_helpers), (transports.WebRiskServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_web_risk_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('webrisk.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='webrisk.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceGrpcAsyncIOTransport])
def test_web_risk_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_web_risk_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.WebRiskServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_web_risk_service_rest_lro_client():
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_web_risk_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='webrisk.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('webrisk.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_web_risk_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='webrisk.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('webrisk.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_web_risk_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = WebRiskServiceClient(credentials=creds1, transport=transport_name)
    client2 = WebRiskServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.compute_threat_list_diff._session
    session2 = client2.transport.compute_threat_list_diff._session
    assert session1 != session2
    session1 = client1.transport.search_uris._session
    session2 = client2.transport.search_uris._session
    assert session1 != session2
    session1 = client1.transport.search_hashes._session
    session2 = client2.transport.search_hashes._session
    assert session1 != session2
    session1 = client1.transport.create_submission._session
    session2 = client2.transport.create_submission._session
    assert session1 != session2
    session1 = client1.transport.submit_uri._session
    session2 = client2.transport.submit_uri._session
    assert session1 != session2

def test_web_risk_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WebRiskServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_web_risk_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WebRiskServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceGrpcAsyncIOTransport])
def test_web_risk_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceGrpcTransport, transports.WebRiskServiceGrpcAsyncIOTransport])
def test_web_risk_service_transport_channel_mtls_with_adc(transport_class):
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

def test_web_risk_service_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_web_risk_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = WebRiskServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = WebRiskServiceClient.common_billing_account_path(**expected)
    actual = WebRiskServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = WebRiskServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'octopus'}
    path = WebRiskServiceClient.common_folder_path(**expected)
    actual = WebRiskServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = WebRiskServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = WebRiskServiceClient.common_organization_path(**expected)
    actual = WebRiskServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = WebRiskServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel'}
    path = WebRiskServiceClient.common_project_path(**expected)
    actual = WebRiskServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = WebRiskServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = WebRiskServiceClient.common_location_path(**expected)
    actual = WebRiskServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.WebRiskServiceTransport, '_prep_wrapped_messages') as prep:
        client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.WebRiskServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = WebRiskServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.CancelOperationRequest, dict])
def test_cancel_operation_rest(request_type):
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_operation(request)
    assert response is None

def test_delete_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.DeleteOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_operation(request)
    assert response is None

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1'}, request)
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
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
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

def test_delete_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = WebRiskServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = WebRiskServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(WebRiskServiceClient, transports.WebRiskServiceGrpcTransport), (WebRiskServiceAsyncClient, transports.WebRiskServiceGrpcAsyncIOTransport)])
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