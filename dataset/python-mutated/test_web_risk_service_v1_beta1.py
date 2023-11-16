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
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.webrisk_v1beta1.services.web_risk_service_v1_beta1 import WebRiskServiceV1Beta1AsyncClient, WebRiskServiceV1Beta1Client, transports
from google.cloud.webrisk_v1beta1.types import webrisk

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
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert WebRiskServiceV1Beta1Client._get_default_mtls_endpoint(None) is None
    assert WebRiskServiceV1Beta1Client._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert WebRiskServiceV1Beta1Client._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert WebRiskServiceV1Beta1Client._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert WebRiskServiceV1Beta1Client._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert WebRiskServiceV1Beta1Client._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(WebRiskServiceV1Beta1Client, 'grpc'), (WebRiskServiceV1Beta1AsyncClient, 'grpc_asyncio'), (WebRiskServiceV1Beta1Client, 'rest')])
def test_web_risk_service_v1_beta1_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('webrisk.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc'), (transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio'), (transports.WebRiskServiceV1Beta1RestTransport, 'rest')])
def test_web_risk_service_v1_beta1_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(WebRiskServiceV1Beta1Client, 'grpc'), (WebRiskServiceV1Beta1AsyncClient, 'grpc_asyncio'), (WebRiskServiceV1Beta1Client, 'rest')])
def test_web_risk_service_v1_beta1_client_from_service_account_file(client_class, transport_name):
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

def test_web_risk_service_v1_beta1_client_get_transport_class():
    if False:
        return 10
    transport = WebRiskServiceV1Beta1Client.get_transport_class()
    available_transports = [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1RestTransport]
    assert transport in available_transports
    transport = WebRiskServiceV1Beta1Client.get_transport_class('grpc')
    assert transport == transports.WebRiskServiceV1Beta1GrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc'), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio'), (WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1RestTransport, 'rest')])
@mock.patch.object(WebRiskServiceV1Beta1Client, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceV1Beta1Client))
@mock.patch.object(WebRiskServiceV1Beta1AsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceV1Beta1AsyncClient))
def test_web_risk_service_v1_beta1_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(WebRiskServiceV1Beta1Client, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(WebRiskServiceV1Beta1Client, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc', 'true'), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc', 'false'), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1RestTransport, 'rest', 'true'), (WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1RestTransport, 'rest', 'false')])
@mock.patch.object(WebRiskServiceV1Beta1Client, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceV1Beta1Client))
@mock.patch.object(WebRiskServiceV1Beta1AsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceV1Beta1AsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_web_risk_service_v1_beta1_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [WebRiskServiceV1Beta1Client, WebRiskServiceV1Beta1AsyncClient])
@mock.patch.object(WebRiskServiceV1Beta1Client, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceV1Beta1Client))
@mock.patch.object(WebRiskServiceV1Beta1AsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WebRiskServiceV1Beta1AsyncClient))
def test_web_risk_service_v1_beta1_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc'), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio'), (WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1RestTransport, 'rest')])
def test_web_risk_service_v1_beta1_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc', grpc_helpers), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1RestTransport, 'rest', None)])
def test_web_risk_service_v1_beta1_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_web_risk_service_v1_beta1_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.webrisk_v1beta1.services.web_risk_service_v1_beta1.transports.WebRiskServiceV1Beta1GrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = WebRiskServiceV1Beta1Client(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport, 'grpc', grpc_helpers), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_web_risk_service_v1_beta1_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('webrisk.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='webrisk.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [webrisk.ComputeThreatListDiffRequest, dict])
def test_compute_threat_list_diff(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.compute_threat_list_diff), '__call__') as call:
        client.compute_threat_list_diff()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.ComputeThreatListDiffRequest()

@pytest.mark.asyncio
async def test_compute_threat_list_diff_async(transport: str='grpc_asyncio', request_type=webrisk.ComputeThreatListDiffRequest):
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.compute_threat_list_diff(webrisk.ComputeThreatListDiffRequest(), threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))

@pytest.mark.asyncio
async def test_compute_threat_list_diff_flattened_async():
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.compute_threat_list_diff(webrisk.ComputeThreatListDiffRequest(), threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))

@pytest.mark.parametrize('request_type', [webrisk.SearchUrisRequest, dict])
def test_search_uris(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_uris), '__call__') as call:
        client.search_uris()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchUrisRequest()

@pytest.mark.asyncio
async def test_search_uris_async(transport: str='grpc_asyncio', request_type=webrisk.SearchUrisRequest):
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_uris(webrisk.SearchUrisRequest(), uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.asyncio
async def test_search_uris_flattened_async():
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_uris(webrisk.SearchUrisRequest(), uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.parametrize('request_type', [webrisk.SearchHashesRequest, dict])
def test_search_hashes(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_hashes), '__call__') as call:
        client.search_hashes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == webrisk.SearchHashesRequest()

@pytest.mark.asyncio
async def test_search_hashes_async(transport: str='grpc_asyncio', request_type=webrisk.SearchHashesRequest):
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_hashes(webrisk.SearchHashesRequest(), hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.asyncio
async def test_search_hashes_flattened_async():
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_hashes(webrisk.SearchHashesRequest(), hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])

@pytest.mark.parametrize('request_type', [webrisk.ComputeThreatListDiffRequest, dict])
def test_compute_threat_list_diff_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport_class = transports.WebRiskServiceV1Beta1RestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).compute_threat_list_diff._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).compute_threat_list_diff._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('constraints', 'threat_type', 'version_token'))
    jsonified_request.update(unset_fields)
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.WebRiskServiceV1Beta1RestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.compute_threat_list_diff._get_unset_required_fields({})
    assert set(unset_fields) == set(('constraints', 'threatType', 'versionToken')) & set(('threatType', 'constraints'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_compute_threat_list_diff_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WebRiskServiceV1Beta1RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceV1Beta1RestInterceptor())
    client = WebRiskServiceV1Beta1Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceV1Beta1RestInterceptor, 'post_compute_threat_list_diff') as post, mock.patch.object(transports.WebRiskServiceV1Beta1RestInterceptor, 'pre_compute_threat_list_diff') as pre:
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
        print('Hello World!')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1beta1/threatLists:computeDiff' % client.transport._host, args[1])

def test_compute_threat_list_diff_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.compute_threat_list_diff(webrisk.ComputeThreatListDiffRequest(), threat_type=webrisk.ThreatType.MALWARE, version_token=b'version_token_blob', constraints=webrisk.ComputeThreatListDiffRequest.Constraints(max_diff_entries=1687))

def test_compute_threat_list_diff_rest_error():
    if False:
        while True:
            i = 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [webrisk.SearchUrisRequest, dict])
def test_search_uris_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport_class = transports.WebRiskServiceV1Beta1RestTransport
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
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.WebRiskServiceV1Beta1RestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_uris._get_unset_required_fields({})
    assert set(unset_fields) == set(('threatTypes', 'uri')) & set(('uri', 'threatTypes'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_uris_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WebRiskServiceV1Beta1RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceV1Beta1RestInterceptor())
    client = WebRiskServiceV1Beta1Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceV1Beta1RestInterceptor, 'post_search_uris') as post, mock.patch.object(transports.WebRiskServiceV1Beta1RestInterceptor, 'pre_search_uris') as pre:
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
        return 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1beta1/uris:search' % client.transport._host, args[1])

def test_search_uris_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_uris(webrisk.SearchUrisRequest(), uri='uri_value', threat_types=[webrisk.ThreatType.MALWARE])

def test_search_uris_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [webrisk.SearchHashesRequest, dict])
def test_search_hashes_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport_class = transports.WebRiskServiceV1Beta1RestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_hashes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_hashes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('hash_prefix', 'threat_types'))
    jsonified_request.update(unset_fields)
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.WebRiskServiceV1Beta1RestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_hashes._get_unset_required_fields({})
    assert set(unset_fields) == set(('hashPrefix', 'threatTypes')) & set(('threatTypes',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_hashes_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WebRiskServiceV1Beta1RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WebRiskServiceV1Beta1RestInterceptor())
    client = WebRiskServiceV1Beta1Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WebRiskServiceV1Beta1RestInterceptor, 'post_search_hashes') as post, mock.patch.object(transports.WebRiskServiceV1Beta1RestInterceptor, 'pre_search_hashes') as pre:
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
        while True:
            i = 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1beta1/hashes:search' % client.transport._host, args[1])

def test_search_hashes_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_hashes(webrisk.SearchHashesRequest(), hash_prefix=b'hash_prefix_blob', threat_types=[webrisk.ThreatType.MALWARE])

def test_search_hashes_rest_error():
    if False:
        print('Hello World!')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WebRiskServiceV1Beta1Client(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WebRiskServiceV1Beta1Client(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WebRiskServiceV1Beta1Client(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WebRiskServiceV1Beta1Client(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = WebRiskServiceV1Beta1Client(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, transports.WebRiskServiceV1Beta1RestTransport])
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
        while True:
            i = 10
    transport = WebRiskServiceV1Beta1Client.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.WebRiskServiceV1Beta1GrpcTransport)

def test_web_risk_service_v1_beta1_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.WebRiskServiceV1Beta1Transport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_web_risk_service_v1_beta1_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.webrisk_v1beta1.services.web_risk_service_v1_beta1.transports.WebRiskServiceV1Beta1Transport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.WebRiskServiceV1Beta1Transport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('compute_threat_list_diff', 'search_uris', 'search_hashes')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_web_risk_service_v1_beta1_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.webrisk_v1beta1.services.web_risk_service_v1_beta1.transports.WebRiskServiceV1Beta1Transport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WebRiskServiceV1Beta1Transport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_web_risk_service_v1_beta1_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.webrisk_v1beta1.services.web_risk_service_v1_beta1.transports.WebRiskServiceV1Beta1Transport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WebRiskServiceV1Beta1Transport()
        adc.assert_called_once()

def test_web_risk_service_v1_beta1_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        WebRiskServiceV1Beta1Client()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport])
def test_web_risk_service_v1_beta1_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, transports.WebRiskServiceV1Beta1RestTransport])
def test_web_risk_service_v1_beta1_transport_auth_gdch_credentials(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.WebRiskServiceV1Beta1GrpcTransport, grpc_helpers), (transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport, grpc_helpers_async)])
def test_web_risk_service_v1_beta1_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('webrisk.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='webrisk.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport])
def test_web_risk_service_v1_beta1_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_web_risk_service_v1_beta1_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.WebRiskServiceV1Beta1RestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_web_risk_service_v1_beta1_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='webrisk.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('webrisk.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_web_risk_service_v1_beta1_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='webrisk.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('webrisk.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://webrisk.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_web_risk_service_v1_beta1_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = WebRiskServiceV1Beta1Client(credentials=creds1, transport=transport_name)
    client2 = WebRiskServiceV1Beta1Client(credentials=creds2, transport=transport_name)
    session1 = client1.transport.compute_threat_list_diff._session
    session2 = client2.transport.compute_threat_list_diff._session
    assert session1 != session2
    session1 = client1.transport.search_uris._session
    session2 = client2.transport.search_uris._session
    assert session1 != session2
    session1 = client1.transport.search_hashes._session
    session2 = client2.transport.search_hashes._session
    assert session1 != session2

def test_web_risk_service_v1_beta1_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WebRiskServiceV1Beta1GrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_web_risk_service_v1_beta1_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport])
def test_web_risk_service_v1_beta1_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class', [transports.WebRiskServiceV1Beta1GrpcTransport, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport])
def test_web_risk_service_v1_beta1_transport_channel_mtls_with_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = WebRiskServiceV1Beta1Client.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'clam'}
    path = WebRiskServiceV1Beta1Client.common_billing_account_path(**expected)
    actual = WebRiskServiceV1Beta1Client.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = WebRiskServiceV1Beta1Client.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = WebRiskServiceV1Beta1Client.common_folder_path(**expected)
    actual = WebRiskServiceV1Beta1Client.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = WebRiskServiceV1Beta1Client.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nudibranch'}
    path = WebRiskServiceV1Beta1Client.common_organization_path(**expected)
    actual = WebRiskServiceV1Beta1Client.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = WebRiskServiceV1Beta1Client.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = WebRiskServiceV1Beta1Client.common_project_path(**expected)
    actual = WebRiskServiceV1Beta1Client.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = WebRiskServiceV1Beta1Client.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = WebRiskServiceV1Beta1Client.common_location_path(**expected)
    actual = WebRiskServiceV1Beta1Client.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.WebRiskServiceV1Beta1Transport, '_prep_wrapped_messages') as prep:
        client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.WebRiskServiceV1Beta1Transport, '_prep_wrapped_messages') as prep:
        transport_class = WebRiskServiceV1Beta1Client.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = WebRiskServiceV1Beta1AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = WebRiskServiceV1Beta1Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(WebRiskServiceV1Beta1Client, transports.WebRiskServiceV1Beta1GrpcTransport), (WebRiskServiceV1Beta1AsyncClient, transports.WebRiskServiceV1Beta1GrpcAsyncIOTransport)])
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