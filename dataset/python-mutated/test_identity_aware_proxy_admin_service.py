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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import wrappers_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.iap_v1.services.identity_aware_proxy_admin_service import IdentityAwareProxyAdminServiceAsyncClient, IdentityAwareProxyAdminServiceClient, pagers, transports
from google.cloud.iap_v1.types import service

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert IdentityAwareProxyAdminServiceClient._get_default_mtls_endpoint(None) is None
    assert IdentityAwareProxyAdminServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert IdentityAwareProxyAdminServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert IdentityAwareProxyAdminServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert IdentityAwareProxyAdminServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert IdentityAwareProxyAdminServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(IdentityAwareProxyAdminServiceClient, 'grpc'), (IdentityAwareProxyAdminServiceAsyncClient, 'grpc_asyncio'), (IdentityAwareProxyAdminServiceClient, 'rest')])
def test_identity_aware_proxy_admin_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('iap.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://iap.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc'), (transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.IdentityAwareProxyAdminServiceRestTransport, 'rest')])
def test_identity_aware_proxy_admin_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(IdentityAwareProxyAdminServiceClient, 'grpc'), (IdentityAwareProxyAdminServiceAsyncClient, 'grpc_asyncio'), (IdentityAwareProxyAdminServiceClient, 'rest')])
def test_identity_aware_proxy_admin_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('iap.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://iap.googleapis.com')

def test_identity_aware_proxy_admin_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = IdentityAwareProxyAdminServiceClient.get_transport_class()
    available_transports = [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceRestTransport]
    assert transport in available_transports
    transport = IdentityAwareProxyAdminServiceClient.get_transport_class('grpc')
    assert transport == transports.IdentityAwareProxyAdminServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc'), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceRestTransport, 'rest')])
@mock.patch.object(IdentityAwareProxyAdminServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IdentityAwareProxyAdminServiceClient))
@mock.patch.object(IdentityAwareProxyAdminServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IdentityAwareProxyAdminServiceAsyncClient))
def test_identity_aware_proxy_admin_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(IdentityAwareProxyAdminServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(IdentityAwareProxyAdminServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc', 'true'), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc', 'false'), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceRestTransport, 'rest', 'true'), (IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceRestTransport, 'rest', 'false')])
@mock.patch.object(IdentityAwareProxyAdminServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IdentityAwareProxyAdminServiceClient))
@mock.patch.object(IdentityAwareProxyAdminServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IdentityAwareProxyAdminServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_identity_aware_proxy_admin_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [IdentityAwareProxyAdminServiceClient, IdentityAwareProxyAdminServiceAsyncClient])
@mock.patch.object(IdentityAwareProxyAdminServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IdentityAwareProxyAdminServiceClient))
@mock.patch.object(IdentityAwareProxyAdminServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IdentityAwareProxyAdminServiceAsyncClient))
def test_identity_aware_proxy_admin_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc'), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceRestTransport, 'rest')])
def test_identity_aware_proxy_admin_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc', grpc_helpers), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceRestTransport, 'rest', None)])
def test_identity_aware_proxy_admin_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_identity_aware_proxy_admin_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.iap_v1.services.identity_aware_proxy_admin_service.transports.IdentityAwareProxyAdminServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = IdentityAwareProxyAdminServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport, 'grpc', grpc_helpers), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_identity_aware_proxy_admin_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('iap.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='iap.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.parametrize('request_type', [service.GetIapSettingsRequest, dict])
def test_get_iap_settings(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iap_settings), '__call__') as call:
        call.return_value = service.IapSettings(name='name_value')
        response = client.get_iap_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetIapSettingsRequest()
    assert isinstance(response, service.IapSettings)
    assert response.name == 'name_value'

def test_get_iap_settings_empty_call():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iap_settings), '__call__') as call:
        client.get_iap_settings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetIapSettingsRequest()

@pytest.mark.asyncio
async def test_get_iap_settings_async(transport: str='grpc_asyncio', request_type=service.GetIapSettingsRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iap_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.IapSettings(name='name_value'))
        response = await client.get_iap_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetIapSettingsRequest()
    assert isinstance(response, service.IapSettings)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_iap_settings_async_from_dict():
    await test_get_iap_settings_async(request_type=dict)

def test_get_iap_settings_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetIapSettingsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_iap_settings), '__call__') as call:
        call.return_value = service.IapSettings()
        client.get_iap_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iap_settings_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetIapSettingsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_iap_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.IapSettings())
        await client.get_iap_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.UpdateIapSettingsRequest, dict])
def test_update_iap_settings(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_iap_settings), '__call__') as call:
        call.return_value = service.IapSettings(name='name_value')
        response = client.update_iap_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateIapSettingsRequest()
    assert isinstance(response, service.IapSettings)
    assert response.name == 'name_value'

def test_update_iap_settings_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_iap_settings), '__call__') as call:
        client.update_iap_settings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateIapSettingsRequest()

@pytest.mark.asyncio
async def test_update_iap_settings_async(transport: str='grpc_asyncio', request_type=service.UpdateIapSettingsRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_iap_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.IapSettings(name='name_value'))
        response = await client.update_iap_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateIapSettingsRequest()
    assert isinstance(response, service.IapSettings)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_iap_settings_async_from_dict():
    await test_update_iap_settings_async(request_type=dict)

def test_update_iap_settings_field_headers():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateIapSettingsRequest()
    request.iap_settings.name = 'name_value'
    with mock.patch.object(type(client.transport.update_iap_settings), '__call__') as call:
        call.return_value = service.IapSettings()
        client.update_iap_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'iap_settings.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_iap_settings_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateIapSettingsRequest()
    request.iap_settings.name = 'name_value'
    with mock.patch.object(type(client.transport.update_iap_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.IapSettings())
        await client.update_iap_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'iap_settings.name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ListTunnelDestGroupsRequest, dict])
def test_list_tunnel_dest_groups(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.return_value = service.ListTunnelDestGroupsResponse(next_page_token='next_page_token_value')
        response = client.list_tunnel_dest_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTunnelDestGroupsRequest()
    assert isinstance(response, pagers.ListTunnelDestGroupsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tunnel_dest_groups_empty_call():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        client.list_tunnel_dest_groups()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTunnelDestGroupsRequest()

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_async(transport: str='grpc_asyncio', request_type=service.ListTunnelDestGroupsRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTunnelDestGroupsResponse(next_page_token='next_page_token_value'))
        response = await client.list_tunnel_dest_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListTunnelDestGroupsRequest()
    assert isinstance(response, pagers.ListTunnelDestGroupsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_async_from_dict():
    await test_list_tunnel_dest_groups_async(request_type=dict)

def test_list_tunnel_dest_groups_field_headers():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListTunnelDestGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.return_value = service.ListTunnelDestGroupsResponse()
        client.list_tunnel_dest_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListTunnelDestGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTunnelDestGroupsResponse())
        await client.list_tunnel_dest_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_tunnel_dest_groups_flattened():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.return_value = service.ListTunnelDestGroupsResponse()
        client.list_tunnel_dest_groups(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tunnel_dest_groups_flattened_error():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tunnel_dest_groups(service.ListTunnelDestGroupsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_flattened_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.return_value = service.ListTunnelDestGroupsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListTunnelDestGroupsResponse())
        response = await client.list_tunnel_dest_groups(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_flattened_error_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tunnel_dest_groups(service.ListTunnelDestGroupsRequest(), parent='parent_value')

def test_list_tunnel_dest_groups_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.side_effect = (service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup(), service.TunnelDestGroup()], next_page_token='abc'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[], next_page_token='def'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup()], next_page_token='ghi'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_tunnel_dest_groups(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.TunnelDestGroup) for i in results))

def test_list_tunnel_dest_groups_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__') as call:
        call.side_effect = (service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup(), service.TunnelDestGroup()], next_page_token='abc'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[], next_page_token='def'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup()], next_page_token='ghi'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup()]), RuntimeError)
        pages = list(client.list_tunnel_dest_groups(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_async_pager():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup(), service.TunnelDestGroup()], next_page_token='abc'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[], next_page_token='def'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup()], next_page_token='ghi'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup()]), RuntimeError)
        async_pager = await client.list_tunnel_dest_groups(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.TunnelDestGroup) for i in responses))

@pytest.mark.asyncio
async def test_list_tunnel_dest_groups_async_pages():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tunnel_dest_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup(), service.TunnelDestGroup()], next_page_token='abc'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[], next_page_token='def'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup()], next_page_token='ghi'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tunnel_dest_groups(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateTunnelDestGroupRequest, dict])
def test_create_tunnel_dest_group(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value'])
        response = client.create_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateTunnelDestGroupRequest()
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

def test_create_tunnel_dest_group_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        client.create_tunnel_dest_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateTunnelDestGroupRequest()

@pytest.mark.asyncio
async def test_create_tunnel_dest_group_async(transport: str='grpc_asyncio', request_type=service.CreateTunnelDestGroupRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value']))
        response = await client.create_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateTunnelDestGroupRequest()
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

@pytest.mark.asyncio
async def test_create_tunnel_dest_group_async_from_dict():
    await test_create_tunnel_dest_group_async(request_type=dict)

def test_create_tunnel_dest_group_field_headers():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateTunnelDestGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        client.create_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_tunnel_dest_group_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateTunnelDestGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup())
        await client.create_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_tunnel_dest_group_flattened():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        client.create_tunnel_dest_group(parent='parent_value', tunnel_dest_group=service.TunnelDestGroup(name='name_value'), tunnel_dest_group_id='tunnel_dest_group_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].tunnel_dest_group
        mock_val = service.TunnelDestGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].tunnel_dest_group_id
        mock_val = 'tunnel_dest_group_id_value'
        assert arg == mock_val

def test_create_tunnel_dest_group_flattened_error():
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_tunnel_dest_group(service.CreateTunnelDestGroupRequest(), parent='parent_value', tunnel_dest_group=service.TunnelDestGroup(name='name_value'), tunnel_dest_group_id='tunnel_dest_group_id_value')

@pytest.mark.asyncio
async def test_create_tunnel_dest_group_flattened_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup())
        response = await client.create_tunnel_dest_group(parent='parent_value', tunnel_dest_group=service.TunnelDestGroup(name='name_value'), tunnel_dest_group_id='tunnel_dest_group_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].tunnel_dest_group
        mock_val = service.TunnelDestGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].tunnel_dest_group_id
        mock_val = 'tunnel_dest_group_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_tunnel_dest_group_flattened_error_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_tunnel_dest_group(service.CreateTunnelDestGroupRequest(), parent='parent_value', tunnel_dest_group=service.TunnelDestGroup(name='name_value'), tunnel_dest_group_id='tunnel_dest_group_id_value')

@pytest.mark.parametrize('request_type', [service.GetTunnelDestGroupRequest, dict])
def test_get_tunnel_dest_group(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value'])
        response = client.get_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetTunnelDestGroupRequest()
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

def test_get_tunnel_dest_group_empty_call():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        client.get_tunnel_dest_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetTunnelDestGroupRequest()

@pytest.mark.asyncio
async def test_get_tunnel_dest_group_async(transport: str='grpc_asyncio', request_type=service.GetTunnelDestGroupRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value']))
        response = await client.get_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetTunnelDestGroupRequest()
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

@pytest.mark.asyncio
async def test_get_tunnel_dest_group_async_from_dict():
    await test_get_tunnel_dest_group_async(request_type=dict)

def test_get_tunnel_dest_group_field_headers():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetTunnelDestGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        client.get_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_tunnel_dest_group_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetTunnelDestGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup())
        await client.get_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_tunnel_dest_group_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        client.get_tunnel_dest_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_tunnel_dest_group_flattened_error():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_tunnel_dest_group(service.GetTunnelDestGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_tunnel_dest_group_flattened_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup())
        response = await client.get_tunnel_dest_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_tunnel_dest_group_flattened_error_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_tunnel_dest_group(service.GetTunnelDestGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteTunnelDestGroupRequest, dict])
def test_delete_tunnel_dest_group(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        call.return_value = None
        response = client.delete_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteTunnelDestGroupRequest()
    assert response is None

def test_delete_tunnel_dest_group_empty_call():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        client.delete_tunnel_dest_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteTunnelDestGroupRequest()

@pytest.mark.asyncio
async def test_delete_tunnel_dest_group_async(transport: str='grpc_asyncio', request_type=service.DeleteTunnelDestGroupRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteTunnelDestGroupRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_tunnel_dest_group_async_from_dict():
    await test_delete_tunnel_dest_group_async(request_type=dict)

def test_delete_tunnel_dest_group_field_headers():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteTunnelDestGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        call.return_value = None
        client.delete_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_tunnel_dest_group_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteTunnelDestGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_tunnel_dest_group_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        call.return_value = None
        client.delete_tunnel_dest_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_tunnel_dest_group_flattened_error():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_tunnel_dest_group(service.DeleteTunnelDestGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_tunnel_dest_group_flattened_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tunnel_dest_group), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_tunnel_dest_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_tunnel_dest_group_flattened_error_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_tunnel_dest_group(service.DeleteTunnelDestGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateTunnelDestGroupRequest, dict])
def test_update_tunnel_dest_group(request_type, transport: str='grpc'):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value'])
        response = client.update_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateTunnelDestGroupRequest()
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

def test_update_tunnel_dest_group_empty_call():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        client.update_tunnel_dest_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateTunnelDestGroupRequest()

@pytest.mark.asyncio
async def test_update_tunnel_dest_group_async(transport: str='grpc_asyncio', request_type=service.UpdateTunnelDestGroupRequest):
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value']))
        response = await client.update_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateTunnelDestGroupRequest()
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

@pytest.mark.asyncio
async def test_update_tunnel_dest_group_async_from_dict():
    await test_update_tunnel_dest_group_async(request_type=dict)

def test_update_tunnel_dest_group_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateTunnelDestGroupRequest()
    request.tunnel_dest_group.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        client.update_tunnel_dest_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tunnel_dest_group.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_tunnel_dest_group_field_headers_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateTunnelDestGroupRequest()
    request.tunnel_dest_group.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup())
        await client.update_tunnel_dest_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tunnel_dest_group.name=name_value') in kw['metadata']

def test_update_tunnel_dest_group_flattened():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        client.update_tunnel_dest_group(tunnel_dest_group=service.TunnelDestGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tunnel_dest_group
        mock_val = service.TunnelDestGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_tunnel_dest_group_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_tunnel_dest_group(service.UpdateTunnelDestGroupRequest(), tunnel_dest_group=service.TunnelDestGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_tunnel_dest_group_flattened_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tunnel_dest_group), '__call__') as call:
        call.return_value = service.TunnelDestGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.TunnelDestGroup())
        response = await client.update_tunnel_dest_group(tunnel_dest_group=service.TunnelDestGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tunnel_dest_group
        mock_val = service.TunnelDestGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_tunnel_dest_group_flattened_error_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_tunnel_dest_group(service.UpdateTunnelDestGroupRequest(), tunnel_dest_group=service.TunnelDestGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_error():
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_error():
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetIapSettingsRequest, dict])
def test_get_iap_settings_rest(request_type):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.IapSettings(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.IapSettings.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iap_settings(request)
    assert isinstance(response, service.IapSettings)
    assert response.name == 'name_value'

def test_get_iap_settings_rest_required_fields(request_type=service.GetIapSettingsRequest):
    if False:
        return 10
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iap_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iap_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.IapSettings()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.IapSettings.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iap_settings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iap_settings_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iap_settings._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iap_settings_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_get_iap_settings') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_get_iap_settings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetIapSettingsRequest.pb(service.GetIapSettingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.IapSettings.to_json(service.IapSettings())
        request = service.GetIapSettingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.IapSettings()
        client.get_iap_settings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iap_settings_rest_bad_request(transport: str='rest', request_type=service.GetIapSettingsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iap_settings(request)

def test_get_iap_settings_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateIapSettingsRequest, dict])
def test_update_iap_settings_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'iap_settings': {'name': 'sample1'}}
    request_init['iap_settings'] = {'name': 'sample1', 'access_settings': {'gcip_settings': {'tenant_ids': ['tenant_ids_value1', 'tenant_ids_value2'], 'login_page_uri': {'value': 'value_value'}}, 'cors_settings': {'allow_http_options': {'value': True}}, 'oauth_settings': {'login_hint': {}, 'programmatic_clients': ['programmatic_clients_value1', 'programmatic_clients_value2']}, 'reauth_settings': {'method': 1, 'max_age': {'seconds': 751, 'nanos': 543}, 'policy_type': 1}, 'allowed_domains_settings': {'enable': True, 'domains': ['domains_value1', 'domains_value2']}}, 'application_settings': {'csm_settings': {'rctoken_aud': {}}, 'access_denied_page_settings': {'access_denied_page_uri': {}, 'generate_troubleshooting_uri': {}, 'remediation_token_generation_enabled': {}}, 'cookie_domain': {}, 'attribute_propagation_settings': {'expression': 'expression_value', 'output_credentials': [1], 'enable': True}}}
    test_field = service.UpdateIapSettingsRequest.meta.fields['iap_settings']

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
    for (field, value) in request_init['iap_settings'].items():
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
                for i in range(0, len(request_init['iap_settings'][field])):
                    del request_init['iap_settings'][field][i][subfield]
            else:
                del request_init['iap_settings'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.IapSettings(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.IapSettings.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_iap_settings(request)
    assert isinstance(response, service.IapSettings)
    assert response.name == 'name_value'

def test_update_iap_settings_rest_required_fields(request_type=service.UpdateIapSettingsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_iap_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_iap_settings._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.IapSettings()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.IapSettings.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_iap_settings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_iap_settings_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_iap_settings._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('iapSettings',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_iap_settings_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_update_iap_settings') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_update_iap_settings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateIapSettingsRequest.pb(service.UpdateIapSettingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.IapSettings.to_json(service.IapSettings())
        request = service.UpdateIapSettingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.IapSettings()
        client.update_iap_settings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_iap_settings_rest_bad_request(transport: str='rest', request_type=service.UpdateIapSettingsRequest):
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'iap_settings': {'name': 'sample1'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_iap_settings(request)

def test_update_iap_settings_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListTunnelDestGroupsRequest, dict])
def test_list_tunnel_dest_groups_rest(request_type):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListTunnelDestGroupsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListTunnelDestGroupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tunnel_dest_groups(request)
    assert isinstance(response, pagers.ListTunnelDestGroupsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tunnel_dest_groups_rest_required_fields(request_type=service.ListTunnelDestGroupsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tunnel_dest_groups._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tunnel_dest_groups._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListTunnelDestGroupsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListTunnelDestGroupsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tunnel_dest_groups(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tunnel_dest_groups_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tunnel_dest_groups._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tunnel_dest_groups_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_list_tunnel_dest_groups') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_list_tunnel_dest_groups') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListTunnelDestGroupsRequest.pb(service.ListTunnelDestGroupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListTunnelDestGroupsResponse.to_json(service.ListTunnelDestGroupsResponse())
        request = service.ListTunnelDestGroupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListTunnelDestGroupsResponse()
        client.list_tunnel_dest_groups(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tunnel_dest_groups_rest_bad_request(transport: str='rest', request_type=service.ListTunnelDestGroupsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tunnel_dest_groups(request)

def test_list_tunnel_dest_groups_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListTunnelDestGroupsResponse()
        sample_request = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListTunnelDestGroupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tunnel_dest_groups(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/iap_tunnel/locations/*}/destGroups' % client.transport._host, args[1])

def test_list_tunnel_dest_groups_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tunnel_dest_groups(service.ListTunnelDestGroupsRequest(), parent='parent_value')

def test_list_tunnel_dest_groups_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup(), service.TunnelDestGroup()], next_page_token='abc'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[], next_page_token='def'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup()], next_page_token='ghi'), service.ListTunnelDestGroupsResponse(tunnel_dest_groups=[service.TunnelDestGroup(), service.TunnelDestGroup()]))
        response = response + response
        response = tuple((service.ListTunnelDestGroupsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
        pager = client.list_tunnel_dest_groups(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.TunnelDestGroup) for i in results))
        pages = list(client.list_tunnel_dest_groups(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateTunnelDestGroupRequest, dict])
def test_create_tunnel_dest_group_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
    request_init['tunnel_dest_group'] = {'name': 'name_value', 'cidrs': ['cidrs_value1', 'cidrs_value2'], 'fqdns': ['fqdns_value1', 'fqdns_value2']}
    test_field = service.CreateTunnelDestGroupRequest.meta.fields['tunnel_dest_group']

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
    for (field, value) in request_init['tunnel_dest_group'].items():
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
                for i in range(0, len(request_init['tunnel_dest_group'][field])):
                    del request_init['tunnel_dest_group'][field][i][subfield]
            else:
                del request_init['tunnel_dest_group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.TunnelDestGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_tunnel_dest_group(request)
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

def test_create_tunnel_dest_group_rest_required_fields(request_type=service.CreateTunnelDestGroupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['tunnel_dest_group_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'tunnelDestGroupId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'tunnelDestGroupId' in jsonified_request
    assert jsonified_request['tunnelDestGroupId'] == request_init['tunnel_dest_group_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['tunnelDestGroupId'] = 'tunnel_dest_group_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('tunnel_dest_group_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'tunnelDestGroupId' in jsonified_request
    assert jsonified_request['tunnelDestGroupId'] == 'tunnel_dest_group_id_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.TunnelDestGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.TunnelDestGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_tunnel_dest_group(request)
            expected_params = [('tunnelDestGroupId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_tunnel_dest_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_tunnel_dest_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('tunnelDestGroupId',)) & set(('parent', 'tunnelDestGroup', 'tunnelDestGroupId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_tunnel_dest_group_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_create_tunnel_dest_group') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_create_tunnel_dest_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateTunnelDestGroupRequest.pb(service.CreateTunnelDestGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.TunnelDestGroup.to_json(service.TunnelDestGroup())
        request = service.CreateTunnelDestGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.TunnelDestGroup()
        client.create_tunnel_dest_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_tunnel_dest_group_rest_bad_request(transport: str='rest', request_type=service.CreateTunnelDestGroupRequest):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_tunnel_dest_group(request)

def test_create_tunnel_dest_group_rest_flattened():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.TunnelDestGroup()
        sample_request = {'parent': 'projects/sample1/iap_tunnel/locations/sample2'}
        mock_args = dict(parent='parent_value', tunnel_dest_group=service.TunnelDestGroup(name='name_value'), tunnel_dest_group_id='tunnel_dest_group_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.TunnelDestGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_tunnel_dest_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/iap_tunnel/locations/*}/destGroups' % client.transport._host, args[1])

def test_create_tunnel_dest_group_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_tunnel_dest_group(service.CreateTunnelDestGroupRequest(), parent='parent_value', tunnel_dest_group=service.TunnelDestGroup(name='name_value'), tunnel_dest_group_id='tunnel_dest_group_id_value')

def test_create_tunnel_dest_group_rest_error():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetTunnelDestGroupRequest, dict])
def test_get_tunnel_dest_group_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.TunnelDestGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_tunnel_dest_group(request)
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

def test_get_tunnel_dest_group_rest_required_fields(request_type=service.GetTunnelDestGroupRequest):
    if False:
        return 10
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.TunnelDestGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.TunnelDestGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_tunnel_dest_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_tunnel_dest_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_tunnel_dest_group._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_tunnel_dest_group_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_get_tunnel_dest_group') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_get_tunnel_dest_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetTunnelDestGroupRequest.pb(service.GetTunnelDestGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.TunnelDestGroup.to_json(service.TunnelDestGroup())
        request = service.GetTunnelDestGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.TunnelDestGroup()
        client.get_tunnel_dest_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_tunnel_dest_group_rest_bad_request(transport: str='rest', request_type=service.GetTunnelDestGroupRequest):
    if False:
        i = 10
        return i + 15
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_tunnel_dest_group(request)

def test_get_tunnel_dest_group_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.TunnelDestGroup()
        sample_request = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.TunnelDestGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_tunnel_dest_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/iap_tunnel/locations/*/destGroups/*}' % client.transport._host, args[1])

def test_get_tunnel_dest_group_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_tunnel_dest_group(service.GetTunnelDestGroupRequest(), name='name_value')

def test_get_tunnel_dest_group_rest_error():
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteTunnelDestGroupRequest, dict])
def test_delete_tunnel_dest_group_rest(request_type):
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_tunnel_dest_group(request)
    assert response is None

def test_delete_tunnel_dest_group_rest_required_fields(request_type=service.DeleteTunnelDestGroupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_tunnel_dest_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_tunnel_dest_group_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_tunnel_dest_group._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_tunnel_dest_group_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_delete_tunnel_dest_group') as pre:
        pre.assert_not_called()
        pb_message = service.DeleteTunnelDestGroupRequest.pb(service.DeleteTunnelDestGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.DeleteTunnelDestGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_tunnel_dest_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_tunnel_dest_group_rest_bad_request(transport: str='rest', request_type=service.DeleteTunnelDestGroupRequest):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_tunnel_dest_group(request)

def test_delete_tunnel_dest_group_rest_flattened():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_tunnel_dest_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/iap_tunnel/locations/*/destGroups/*}' % client.transport._host, args[1])

def test_delete_tunnel_dest_group_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_tunnel_dest_group(service.DeleteTunnelDestGroupRequest(), name='name_value')

def test_delete_tunnel_dest_group_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateTunnelDestGroupRequest, dict])
def test_update_tunnel_dest_group_rest(request_type):
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'tunnel_dest_group': {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}}
    request_init['tunnel_dest_group'] = {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3', 'cidrs': ['cidrs_value1', 'cidrs_value2'], 'fqdns': ['fqdns_value1', 'fqdns_value2']}
    test_field = service.UpdateTunnelDestGroupRequest.meta.fields['tunnel_dest_group']

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
    for (field, value) in request_init['tunnel_dest_group'].items():
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
                for i in range(0, len(request_init['tunnel_dest_group'][field])):
                    del request_init['tunnel_dest_group'][field][i][subfield]
            else:
                del request_init['tunnel_dest_group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.TunnelDestGroup(name='name_value', cidrs=['cidrs_value'], fqdns=['fqdns_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.TunnelDestGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_tunnel_dest_group(request)
    assert isinstance(response, service.TunnelDestGroup)
    assert response.name == 'name_value'
    assert response.cidrs == ['cidrs_value']
    assert response.fqdns == ['fqdns_value']

def test_update_tunnel_dest_group_rest_required_fields(request_type=service.UpdateTunnelDestGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.IdentityAwareProxyAdminServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tunnel_dest_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.TunnelDestGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.TunnelDestGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_tunnel_dest_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_tunnel_dest_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_tunnel_dest_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('tunnelDestGroup',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_tunnel_dest_group_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.IdentityAwareProxyAdminServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IdentityAwareProxyAdminServiceRestInterceptor())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'post_update_tunnel_dest_group') as post, mock.patch.object(transports.IdentityAwareProxyAdminServiceRestInterceptor, 'pre_update_tunnel_dest_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateTunnelDestGroupRequest.pb(service.UpdateTunnelDestGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.TunnelDestGroup.to_json(service.TunnelDestGroup())
        request = service.UpdateTunnelDestGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.TunnelDestGroup()
        client.update_tunnel_dest_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_tunnel_dest_group_rest_bad_request(transport: str='rest', request_type=service.UpdateTunnelDestGroupRequest):
    if False:
        print('Hello World!')
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'tunnel_dest_group': {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_tunnel_dest_group(request)

def test_update_tunnel_dest_group_rest_flattened():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.TunnelDestGroup()
        sample_request = {'tunnel_dest_group': {'name': 'projects/sample1/iap_tunnel/locations/sample2/destGroups/sample3'}}
        mock_args = dict(tunnel_dest_group=service.TunnelDestGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.TunnelDestGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_tunnel_dest_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{tunnel_dest_group.name=projects/*/iap_tunnel/locations/*/destGroups/*}' % client.transport._host, args[1])

def test_update_tunnel_dest_group_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_tunnel_dest_group(service.UpdateTunnelDestGroupRequest(), tunnel_dest_group=service.TunnelDestGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_tunnel_dest_group_rest_error():
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = IdentityAwareProxyAdminServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = IdentityAwareProxyAdminServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = IdentityAwareProxyAdminServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = IdentityAwareProxyAdminServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = IdentityAwareProxyAdminServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, transports.IdentityAwareProxyAdminServiceRestTransport])
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
    transport = IdentityAwareProxyAdminServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.IdentityAwareProxyAdminServiceGrpcTransport)

def test_identity_aware_proxy_admin_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.IdentityAwareProxyAdminServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_identity_aware_proxy_admin_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.iap_v1.services.identity_aware_proxy_admin_service.transports.IdentityAwareProxyAdminServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.IdentityAwareProxyAdminServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_iap_settings', 'update_iap_settings', 'list_tunnel_dest_groups', 'create_tunnel_dest_group', 'get_tunnel_dest_group', 'delete_tunnel_dest_group', 'update_tunnel_dest_group')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_identity_aware_proxy_admin_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.iap_v1.services.identity_aware_proxy_admin_service.transports.IdentityAwareProxyAdminServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.IdentityAwareProxyAdminServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_identity_aware_proxy_admin_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.iap_v1.services.identity_aware_proxy_admin_service.transports.IdentityAwareProxyAdminServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.IdentityAwareProxyAdminServiceTransport()
        adc.assert_called_once()

def test_identity_aware_proxy_admin_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        IdentityAwareProxyAdminServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport])
def test_identity_aware_proxy_admin_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, transports.IdentityAwareProxyAdminServiceRestTransport])
def test_identity_aware_proxy_admin_service_transport_auth_gdch_credentials(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.IdentityAwareProxyAdminServiceGrpcTransport, grpc_helpers), (transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_identity_aware_proxy_admin_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('iap.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='iap.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport])
def test_identity_aware_proxy_admin_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_identity_aware_proxy_admin_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.IdentityAwareProxyAdminServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_identity_aware_proxy_admin_service_host_no_port(transport_name):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='iap.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('iap.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://iap.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_identity_aware_proxy_admin_service_host_with_port(transport_name):
    if False:
        return 10
    client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='iap.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('iap.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://iap.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_identity_aware_proxy_admin_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = IdentityAwareProxyAdminServiceClient(credentials=creds1, transport=transport_name)
    client2 = IdentityAwareProxyAdminServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2
    session1 = client1.transport.get_iap_settings._session
    session2 = client2.transport.get_iap_settings._session
    assert session1 != session2
    session1 = client1.transport.update_iap_settings._session
    session2 = client2.transport.update_iap_settings._session
    assert session1 != session2
    session1 = client1.transport.list_tunnel_dest_groups._session
    session2 = client2.transport.list_tunnel_dest_groups._session
    assert session1 != session2
    session1 = client1.transport.create_tunnel_dest_group._session
    session2 = client2.transport.create_tunnel_dest_group._session
    assert session1 != session2
    session1 = client1.transport.get_tunnel_dest_group._session
    session2 = client2.transport.get_tunnel_dest_group._session
    assert session1 != session2
    session1 = client1.transport.delete_tunnel_dest_group._session
    session2 = client2.transport.delete_tunnel_dest_group._session
    assert session1 != session2
    session1 = client1.transport.update_tunnel_dest_group._session
    session2 = client2.transport.update_tunnel_dest_group._session
    assert session1 != session2

def test_identity_aware_proxy_admin_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.IdentityAwareProxyAdminServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_identity_aware_proxy_admin_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport])
def test_identity_aware_proxy_admin_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.IdentityAwareProxyAdminServiceGrpcTransport, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport])
def test_identity_aware_proxy_admin_service_transport_channel_mtls_with_adc(transport_class):
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

def test_tunnel_dest_group_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    dest_group = 'whelk'
    expected = 'projects/{project}/iap_tunnel/locations/{location}/destGroups/{dest_group}'.format(project=project, location=location, dest_group=dest_group)
    actual = IdentityAwareProxyAdminServiceClient.tunnel_dest_group_path(project, location, dest_group)
    assert expected == actual

def test_parse_tunnel_dest_group_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'dest_group': 'nudibranch'}
    path = IdentityAwareProxyAdminServiceClient.tunnel_dest_group_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_tunnel_dest_group_path(path)
    assert expected == actual

def test_tunnel_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/iap_tunnel/locations/{location}'.format(project=project, location=location)
    actual = IdentityAwareProxyAdminServiceClient.tunnel_location_path(project, location)
    assert expected == actual

def test_parse_tunnel_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = IdentityAwareProxyAdminServiceClient.tunnel_location_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_tunnel_location_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = IdentityAwareProxyAdminServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'abalone'}
    path = IdentityAwareProxyAdminServiceClient.common_billing_account_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = IdentityAwareProxyAdminServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'clam'}
    path = IdentityAwareProxyAdminServiceClient.common_folder_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = IdentityAwareProxyAdminServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'octopus'}
    path = IdentityAwareProxyAdminServiceClient.common_organization_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = IdentityAwareProxyAdminServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch'}
    path = IdentityAwareProxyAdminServiceClient.common_project_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = IdentityAwareProxyAdminServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = IdentityAwareProxyAdminServiceClient.common_location_path(**expected)
    actual = IdentityAwareProxyAdminServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.IdentityAwareProxyAdminServiceTransport, '_prep_wrapped_messages') as prep:
        client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.IdentityAwareProxyAdminServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = IdentityAwareProxyAdminServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = IdentityAwareProxyAdminServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = IdentityAwareProxyAdminServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(IdentityAwareProxyAdminServiceClient, transports.IdentityAwareProxyAdminServiceGrpcTransport), (IdentityAwareProxyAdminServiceAsyncClient, transports.IdentityAwareProxyAdminServiceGrpcAsyncIOTransport)])
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