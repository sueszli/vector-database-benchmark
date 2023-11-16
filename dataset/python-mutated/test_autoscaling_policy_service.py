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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dataproc_v1.services.autoscaling_policy_service import AutoscalingPolicyServiceAsyncClient, AutoscalingPolicyServiceClient, pagers, transports
from google.cloud.dataproc_v1.types import autoscaling_policies

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert AutoscalingPolicyServiceClient._get_default_mtls_endpoint(None) is None
    assert AutoscalingPolicyServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AutoscalingPolicyServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AutoscalingPolicyServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AutoscalingPolicyServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AutoscalingPolicyServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AutoscalingPolicyServiceClient, 'grpc'), (AutoscalingPolicyServiceAsyncClient, 'grpc_asyncio'), (AutoscalingPolicyServiceClient, 'rest')])
def test_autoscaling_policy_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AutoscalingPolicyServiceGrpcTransport, 'grpc'), (transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AutoscalingPolicyServiceRestTransport, 'rest')])
def test_autoscaling_policy_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AutoscalingPolicyServiceClient, 'grpc'), (AutoscalingPolicyServiceAsyncClient, 'grpc_asyncio'), (AutoscalingPolicyServiceClient, 'rest')])
def test_autoscaling_policy_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

def test_autoscaling_policy_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = AutoscalingPolicyServiceClient.get_transport_class()
    available_transports = [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceRestTransport]
    assert transport in available_transports
    transport = AutoscalingPolicyServiceClient.get_transport_class('grpc')
    assert transport == transports.AutoscalingPolicyServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport, 'grpc'), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceRestTransport, 'rest')])
@mock.patch.object(AutoscalingPolicyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AutoscalingPolicyServiceClient))
@mock.patch.object(AutoscalingPolicyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AutoscalingPolicyServiceAsyncClient))
def test_autoscaling_policy_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(AutoscalingPolicyServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AutoscalingPolicyServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport, 'grpc', 'true'), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport, 'grpc', 'false'), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceRestTransport, 'rest', 'true'), (AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceRestTransport, 'rest', 'false')])
@mock.patch.object(AutoscalingPolicyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AutoscalingPolicyServiceClient))
@mock.patch.object(AutoscalingPolicyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AutoscalingPolicyServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_autoscaling_policy_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AutoscalingPolicyServiceClient, AutoscalingPolicyServiceAsyncClient])
@mock.patch.object(AutoscalingPolicyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AutoscalingPolicyServiceClient))
@mock.patch.object(AutoscalingPolicyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AutoscalingPolicyServiceAsyncClient))
def test_autoscaling_policy_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport, 'grpc'), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceRestTransport, 'rest')])
def test_autoscaling_policy_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport, 'grpc', grpc_helpers), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceRestTransport, 'rest', None)])
def test_autoscaling_policy_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_autoscaling_policy_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.dataproc_v1.services.autoscaling_policy_service.transports.AutoscalingPolicyServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AutoscalingPolicyServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport, 'grpc', grpc_helpers), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_autoscaling_policy_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [autoscaling_policies.CreateAutoscalingPolicyRequest, dict])
def test_create_autoscaling_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value')
        response = client.create_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.CreateAutoscalingPolicyRequest()
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

def test_create_autoscaling_policy_empty_call():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        client.create_autoscaling_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.CreateAutoscalingPolicyRequest()

@pytest.mark.asyncio
async def test_create_autoscaling_policy_async(transport: str='grpc_asyncio', request_type=autoscaling_policies.CreateAutoscalingPolicyRequest):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value'))
        response = await client.create_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.CreateAutoscalingPolicyRequest()
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_autoscaling_policy_async_from_dict():
    await test_create_autoscaling_policy_async(request_type=dict)

def test_create_autoscaling_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.CreateAutoscalingPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        client.create_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_autoscaling_policy_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.CreateAutoscalingPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy())
        await client.create_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_autoscaling_policy_flattened():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        client.create_autoscaling_policy(parent='parent_value', policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].policy
        mock_val = autoscaling_policies.AutoscalingPolicy(id='id_value')
        assert arg == mock_val

def test_create_autoscaling_policy_flattened_error():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_autoscaling_policy(autoscaling_policies.CreateAutoscalingPolicyRequest(), parent='parent_value', policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))

@pytest.mark.asyncio
async def test_create_autoscaling_policy_flattened_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy())
        response = await client.create_autoscaling_policy(parent='parent_value', policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].policy
        mock_val = autoscaling_policies.AutoscalingPolicy(id='id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_autoscaling_policy_flattened_error_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_autoscaling_policy(autoscaling_policies.CreateAutoscalingPolicyRequest(), parent='parent_value', policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))

@pytest.mark.parametrize('request_type', [autoscaling_policies.UpdateAutoscalingPolicyRequest, dict])
def test_update_autoscaling_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value')
        response = client.update_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.UpdateAutoscalingPolicyRequest()
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

def test_update_autoscaling_policy_empty_call():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        client.update_autoscaling_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.UpdateAutoscalingPolicyRequest()

@pytest.mark.asyncio
async def test_update_autoscaling_policy_async(transport: str='grpc_asyncio', request_type=autoscaling_policies.UpdateAutoscalingPolicyRequest):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value'))
        response = await client.update_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.UpdateAutoscalingPolicyRequest()
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_autoscaling_policy_async_from_dict():
    await test_update_autoscaling_policy_async(request_type=dict)

def test_update_autoscaling_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.UpdateAutoscalingPolicyRequest()
    request.policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        client.update_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_autoscaling_policy_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.UpdateAutoscalingPolicyRequest()
    request.policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy())
        await client.update_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'policy.name=name_value') in kw['metadata']

def test_update_autoscaling_policy_flattened():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        client.update_autoscaling_policy(policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].policy
        mock_val = autoscaling_policies.AutoscalingPolicy(id='id_value')
        assert arg == mock_val

def test_update_autoscaling_policy_flattened_error():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_autoscaling_policy(autoscaling_policies.UpdateAutoscalingPolicyRequest(), policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))

@pytest.mark.asyncio
async def test_update_autoscaling_policy_flattened_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy())
        response = await client.update_autoscaling_policy(policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].policy
        mock_val = autoscaling_policies.AutoscalingPolicy(id='id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_autoscaling_policy_flattened_error_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_autoscaling_policy(autoscaling_policies.UpdateAutoscalingPolicyRequest(), policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))

@pytest.mark.parametrize('request_type', [autoscaling_policies.GetAutoscalingPolicyRequest, dict])
def test_get_autoscaling_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value')
        response = client.get_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.GetAutoscalingPolicyRequest()
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

def test_get_autoscaling_policy_empty_call():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        client.get_autoscaling_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.GetAutoscalingPolicyRequest()

@pytest.mark.asyncio
async def test_get_autoscaling_policy_async(transport: str='grpc_asyncio', request_type=autoscaling_policies.GetAutoscalingPolicyRequest):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value'))
        response = await client.get_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.GetAutoscalingPolicyRequest()
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_autoscaling_policy_async_from_dict():
    await test_get_autoscaling_policy_async(request_type=dict)

def test_get_autoscaling_policy_field_headers():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.GetAutoscalingPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        client.get_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_autoscaling_policy_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.GetAutoscalingPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy())
        await client.get_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_autoscaling_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        client.get_autoscaling_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_autoscaling_policy_flattened_error():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_autoscaling_policy(autoscaling_policies.GetAutoscalingPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_autoscaling_policy_flattened_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_autoscaling_policy), '__call__') as call:
        call.return_value = autoscaling_policies.AutoscalingPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.AutoscalingPolicy())
        response = await client.get_autoscaling_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_autoscaling_policy_flattened_error_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_autoscaling_policy(autoscaling_policies.GetAutoscalingPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [autoscaling_policies.ListAutoscalingPoliciesRequest, dict])
def test_list_autoscaling_policies(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.return_value = autoscaling_policies.ListAutoscalingPoliciesResponse(next_page_token='next_page_token_value')
        response = client.list_autoscaling_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.ListAutoscalingPoliciesRequest()
    assert isinstance(response, pagers.ListAutoscalingPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_autoscaling_policies_empty_call():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        client.list_autoscaling_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.ListAutoscalingPoliciesRequest()

@pytest.mark.asyncio
async def test_list_autoscaling_policies_async(transport: str='grpc_asyncio', request_type=autoscaling_policies.ListAutoscalingPoliciesRequest):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.ListAutoscalingPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.list_autoscaling_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.ListAutoscalingPoliciesRequest()
    assert isinstance(response, pagers.ListAutoscalingPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_autoscaling_policies_async_from_dict():
    await test_list_autoscaling_policies_async(request_type=dict)

def test_list_autoscaling_policies_field_headers():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.ListAutoscalingPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.return_value = autoscaling_policies.ListAutoscalingPoliciesResponse()
        client.list_autoscaling_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_autoscaling_policies_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.ListAutoscalingPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.ListAutoscalingPoliciesResponse())
        await client.list_autoscaling_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_autoscaling_policies_flattened():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.return_value = autoscaling_policies.ListAutoscalingPoliciesResponse()
        client.list_autoscaling_policies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_autoscaling_policies_flattened_error():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_autoscaling_policies(autoscaling_policies.ListAutoscalingPoliciesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_autoscaling_policies_flattened_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.return_value = autoscaling_policies.ListAutoscalingPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(autoscaling_policies.ListAutoscalingPoliciesResponse())
        response = await client.list_autoscaling_policies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_autoscaling_policies_flattened_error_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_autoscaling_policies(autoscaling_policies.ListAutoscalingPoliciesRequest(), parent='parent_value')

def test_list_autoscaling_policies_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.side_effect = (autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()], next_page_token='abc'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[], next_page_token='def'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy()], next_page_token='ghi'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_autoscaling_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, autoscaling_policies.AutoscalingPolicy) for i in results))

def test_list_autoscaling_policies_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__') as call:
        call.side_effect = (autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()], next_page_token='abc'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[], next_page_token='def'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy()], next_page_token='ghi'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()]), RuntimeError)
        pages = list(client.list_autoscaling_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_autoscaling_policies_async_pager():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()], next_page_token='abc'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[], next_page_token='def'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy()], next_page_token='ghi'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()]), RuntimeError)
        async_pager = await client.list_autoscaling_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, autoscaling_policies.AutoscalingPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_autoscaling_policies_async_pages():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_autoscaling_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()], next_page_token='abc'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[], next_page_token='def'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy()], next_page_token='ghi'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_autoscaling_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [autoscaling_policies.DeleteAutoscalingPolicyRequest, dict])
def test_delete_autoscaling_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        call.return_value = None
        response = client.delete_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.DeleteAutoscalingPolicyRequest()
    assert response is None

def test_delete_autoscaling_policy_empty_call():
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        client.delete_autoscaling_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.DeleteAutoscalingPolicyRequest()

@pytest.mark.asyncio
async def test_delete_autoscaling_policy_async(transport: str='grpc_asyncio', request_type=autoscaling_policies.DeleteAutoscalingPolicyRequest):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == autoscaling_policies.DeleteAutoscalingPolicyRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_autoscaling_policy_async_from_dict():
    await test_delete_autoscaling_policy_async(request_type=dict)

def test_delete_autoscaling_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.DeleteAutoscalingPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        call.return_value = None
        client.delete_autoscaling_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_autoscaling_policy_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = autoscaling_policies.DeleteAutoscalingPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_autoscaling_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_autoscaling_policy_flattened():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        call.return_value = None
        client.delete_autoscaling_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_autoscaling_policy_flattened_error():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_autoscaling_policy(autoscaling_policies.DeleteAutoscalingPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_autoscaling_policy_flattened_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_autoscaling_policy), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_autoscaling_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_autoscaling_policy_flattened_error_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_autoscaling_policy(autoscaling_policies.DeleteAutoscalingPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [autoscaling_policies.CreateAutoscalingPolicyRequest, dict])
def test_create_autoscaling_policy_rest(request_type):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['policy'] = {'id': 'id_value', 'name': 'name_value', 'basic_algorithm': {'yarn_config': {'graceful_decommission_timeout': {'seconds': 751, 'nanos': 543}, 'scale_up_factor': 0.1578, 'scale_down_factor': 0.1789, 'scale_up_min_worker_fraction': 0.2973, 'scale_down_min_worker_fraction': 0.3184}, 'cooldown_period': {}}, 'worker_config': {'min_instances': 1387, 'max_instances': 1389, 'weight': 648}, 'secondary_worker_config': {}, 'labels': {}}
    test_field = autoscaling_policies.CreateAutoscalingPolicyRequest.meta.fields['policy']

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
    for (field, value) in request_init['policy'].items():
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
                for i in range(0, len(request_init['policy'][field])):
                    del request_init['policy'][field][i][subfield]
            else:
                del request_init['policy'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_autoscaling_policy(request)
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

def test_create_autoscaling_policy_rest_required_fields(request_type=autoscaling_policies.CreateAutoscalingPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AutoscalingPolicyServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = autoscaling_policies.AutoscalingPolicy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_autoscaling_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_autoscaling_policy_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_autoscaling_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_autoscaling_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AutoscalingPolicyServiceRestInterceptor())
    client = AutoscalingPolicyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'post_create_autoscaling_policy') as post, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'pre_create_autoscaling_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = autoscaling_policies.CreateAutoscalingPolicyRequest.pb(autoscaling_policies.CreateAutoscalingPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = autoscaling_policies.AutoscalingPolicy.to_json(autoscaling_policies.AutoscalingPolicy())
        request = autoscaling_policies.CreateAutoscalingPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = autoscaling_policies.AutoscalingPolicy()
        client.create_autoscaling_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_autoscaling_policy_rest_bad_request(transport: str='rest', request_type=autoscaling_policies.CreateAutoscalingPolicyRequest):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_autoscaling_policy(request)

def test_create_autoscaling_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.AutoscalingPolicy()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_autoscaling_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/autoscalingPolicies' % client.transport._host, args[1])

def test_create_autoscaling_policy_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_autoscaling_policy(autoscaling_policies.CreateAutoscalingPolicyRequest(), parent='parent_value', policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))

def test_create_autoscaling_policy_rest_error():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [autoscaling_policies.UpdateAutoscalingPolicyRequest, dict])
def test_update_autoscaling_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'policy': {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}}
    request_init['policy'] = {'id': 'id_value', 'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3', 'basic_algorithm': {'yarn_config': {'graceful_decommission_timeout': {'seconds': 751, 'nanos': 543}, 'scale_up_factor': 0.1578, 'scale_down_factor': 0.1789, 'scale_up_min_worker_fraction': 0.2973, 'scale_down_min_worker_fraction': 0.3184}, 'cooldown_period': {}}, 'worker_config': {'min_instances': 1387, 'max_instances': 1389, 'weight': 648}, 'secondary_worker_config': {}, 'labels': {}}
    test_field = autoscaling_policies.UpdateAutoscalingPolicyRequest.meta.fields['policy']

    def get_message_fields(field):
        if False:
            for i in range(10):
                print('nop')
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
    for (field, value) in request_init['policy'].items():
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
                for i in range(0, len(request_init['policy'][field])):
                    del request_init['policy'][field][i][subfield]
            else:
                del request_init['policy'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_autoscaling_policy(request)
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

def test_update_autoscaling_policy_rest_required_fields(request_type=autoscaling_policies.UpdateAutoscalingPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AutoscalingPolicyServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = autoscaling_policies.AutoscalingPolicy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'put', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_autoscaling_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_autoscaling_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_autoscaling_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('policy',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_autoscaling_policy_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AutoscalingPolicyServiceRestInterceptor())
    client = AutoscalingPolicyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'post_update_autoscaling_policy') as post, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'pre_update_autoscaling_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = autoscaling_policies.UpdateAutoscalingPolicyRequest.pb(autoscaling_policies.UpdateAutoscalingPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = autoscaling_policies.AutoscalingPolicy.to_json(autoscaling_policies.AutoscalingPolicy())
        request = autoscaling_policies.UpdateAutoscalingPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = autoscaling_policies.AutoscalingPolicy()
        client.update_autoscaling_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_autoscaling_policy_rest_bad_request(transport: str='rest', request_type=autoscaling_policies.UpdateAutoscalingPolicyRequest):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'policy': {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_autoscaling_policy(request)

def test_update_autoscaling_policy_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.AutoscalingPolicy()
        sample_request = {'policy': {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}}
        mock_args = dict(policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_autoscaling_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{policy.name=projects/*/locations/*/autoscalingPolicies/*}' % client.transport._host, args[1])

def test_update_autoscaling_policy_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_autoscaling_policy(autoscaling_policies.UpdateAutoscalingPolicyRequest(), policy=autoscaling_policies.AutoscalingPolicy(id='id_value'))

def test_update_autoscaling_policy_rest_error():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [autoscaling_policies.GetAutoscalingPolicyRequest, dict])
def test_get_autoscaling_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.AutoscalingPolicy(id='id_value', name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_autoscaling_policy(request)
    assert isinstance(response, autoscaling_policies.AutoscalingPolicy)
    assert response.id == 'id_value'
    assert response.name == 'name_value'

def test_get_autoscaling_policy_rest_required_fields(request_type=autoscaling_policies.GetAutoscalingPolicyRequest):
    if False:
        return 10
    transport_class = transports.AutoscalingPolicyServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = autoscaling_policies.AutoscalingPolicy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_autoscaling_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_autoscaling_policy_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_autoscaling_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_autoscaling_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AutoscalingPolicyServiceRestInterceptor())
    client = AutoscalingPolicyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'post_get_autoscaling_policy') as post, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'pre_get_autoscaling_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = autoscaling_policies.GetAutoscalingPolicyRequest.pb(autoscaling_policies.GetAutoscalingPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = autoscaling_policies.AutoscalingPolicy.to_json(autoscaling_policies.AutoscalingPolicy())
        request = autoscaling_policies.GetAutoscalingPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = autoscaling_policies.AutoscalingPolicy()
        client.get_autoscaling_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_autoscaling_policy_rest_bad_request(transport: str='rest', request_type=autoscaling_policies.GetAutoscalingPolicyRequest):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_autoscaling_policy(request)

def test_get_autoscaling_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.AutoscalingPolicy()
        sample_request = {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.AutoscalingPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_autoscaling_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/autoscalingPolicies/*}' % client.transport._host, args[1])

def test_get_autoscaling_policy_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_autoscaling_policy(autoscaling_policies.GetAutoscalingPolicyRequest(), name='name_value')

def test_get_autoscaling_policy_rest_error():
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [autoscaling_policies.ListAutoscalingPoliciesRequest, dict])
def test_list_autoscaling_policies_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.ListAutoscalingPoliciesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.ListAutoscalingPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_autoscaling_policies(request)
    assert isinstance(response, pagers.ListAutoscalingPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_autoscaling_policies_rest_required_fields(request_type=autoscaling_policies.ListAutoscalingPoliciesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AutoscalingPolicyServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_autoscaling_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_autoscaling_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = autoscaling_policies.ListAutoscalingPoliciesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = autoscaling_policies.ListAutoscalingPoliciesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_autoscaling_policies(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_autoscaling_policies_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_autoscaling_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_autoscaling_policies_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AutoscalingPolicyServiceRestInterceptor())
    client = AutoscalingPolicyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'post_list_autoscaling_policies') as post, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'pre_list_autoscaling_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = autoscaling_policies.ListAutoscalingPoliciesRequest.pb(autoscaling_policies.ListAutoscalingPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = autoscaling_policies.ListAutoscalingPoliciesResponse.to_json(autoscaling_policies.ListAutoscalingPoliciesResponse())
        request = autoscaling_policies.ListAutoscalingPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = autoscaling_policies.ListAutoscalingPoliciesResponse()
        client.list_autoscaling_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_autoscaling_policies_rest_bad_request(transport: str='rest', request_type=autoscaling_policies.ListAutoscalingPoliciesRequest):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_autoscaling_policies(request)

def test_list_autoscaling_policies_rest_flattened():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = autoscaling_policies.ListAutoscalingPoliciesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = autoscaling_policies.ListAutoscalingPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_autoscaling_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/autoscalingPolicies' % client.transport._host, args[1])

def test_list_autoscaling_policies_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_autoscaling_policies(autoscaling_policies.ListAutoscalingPoliciesRequest(), parent='parent_value')

def test_list_autoscaling_policies_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()], next_page_token='abc'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[], next_page_token='def'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy()], next_page_token='ghi'), autoscaling_policies.ListAutoscalingPoliciesResponse(policies=[autoscaling_policies.AutoscalingPolicy(), autoscaling_policies.AutoscalingPolicy()]))
        response = response + response
        response = tuple((autoscaling_policies.ListAutoscalingPoliciesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_autoscaling_policies(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, autoscaling_policies.AutoscalingPolicy) for i in results))
        pages = list(client.list_autoscaling_policies(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [autoscaling_policies.DeleteAutoscalingPolicyRequest, dict])
def test_delete_autoscaling_policy_rest(request_type):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_autoscaling_policy(request)
    assert response is None

def test_delete_autoscaling_policy_rest_required_fields(request_type=autoscaling_policies.DeleteAutoscalingPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AutoscalingPolicyServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_autoscaling_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_autoscaling_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_autoscaling_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_autoscaling_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_autoscaling_policy_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AutoscalingPolicyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AutoscalingPolicyServiceRestInterceptor())
    client = AutoscalingPolicyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AutoscalingPolicyServiceRestInterceptor, 'pre_delete_autoscaling_policy') as pre:
        pre.assert_not_called()
        pb_message = autoscaling_policies.DeleteAutoscalingPolicyRequest.pb(autoscaling_policies.DeleteAutoscalingPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = autoscaling_policies.DeleteAutoscalingPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_autoscaling_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_autoscaling_policy_rest_bad_request(transport: str='rest', request_type=autoscaling_policies.DeleteAutoscalingPolicyRequest):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_autoscaling_policy(request)

def test_delete_autoscaling_policy_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/autoscalingPolicies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_autoscaling_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/autoscalingPolicies/*}' % client.transport._host, args[1])

def test_delete_autoscaling_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_autoscaling_policy(autoscaling_policies.DeleteAutoscalingPolicyRequest(), name='name_value')

def test_delete_autoscaling_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AutoscalingPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AutoscalingPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AutoscalingPolicyServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AutoscalingPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AutoscalingPolicyServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AutoscalingPolicyServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AutoscalingPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AutoscalingPolicyServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.AutoscalingPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AutoscalingPolicyServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.AutoscalingPolicyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AutoscalingPolicyServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, transports.AutoscalingPolicyServiceRestTransport])
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
    transport = AutoscalingPolicyServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AutoscalingPolicyServiceGrpcTransport)

def test_autoscaling_policy_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AutoscalingPolicyServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_autoscaling_policy_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dataproc_v1.services.autoscaling_policy_service.transports.AutoscalingPolicyServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AutoscalingPolicyServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_autoscaling_policy', 'update_autoscaling_policy', 'get_autoscaling_policy', 'list_autoscaling_policies', 'delete_autoscaling_policy', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_autoscaling_policy_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataproc_v1.services.autoscaling_policy_service.transports.AutoscalingPolicyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AutoscalingPolicyServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_autoscaling_policy_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataproc_v1.services.autoscaling_policy_service.transports.AutoscalingPolicyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AutoscalingPolicyServiceTransport()
        adc.assert_called_once()

def test_autoscaling_policy_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AutoscalingPolicyServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport])
def test_autoscaling_policy_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, transports.AutoscalingPolicyServiceRestTransport])
def test_autoscaling_policy_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AutoscalingPolicyServiceGrpcTransport, grpc_helpers), (transports.AutoscalingPolicyServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_autoscaling_policy_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport])
def test_autoscaling_policy_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_autoscaling_policy_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AutoscalingPolicyServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_autoscaling_policy_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_autoscaling_policy_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_autoscaling_policy_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AutoscalingPolicyServiceClient(credentials=creds1, transport=transport_name)
    client2 = AutoscalingPolicyServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_autoscaling_policy._session
    session2 = client2.transport.create_autoscaling_policy._session
    assert session1 != session2
    session1 = client1.transport.update_autoscaling_policy._session
    session2 = client2.transport.update_autoscaling_policy._session
    assert session1 != session2
    session1 = client1.transport.get_autoscaling_policy._session
    session2 = client2.transport.get_autoscaling_policy._session
    assert session1 != session2
    session1 = client1.transport.list_autoscaling_policies._session
    session2 = client2.transport.list_autoscaling_policies._session
    assert session1 != session2
    session1 = client1.transport.delete_autoscaling_policy._session
    session2 = client2.transport.delete_autoscaling_policy._session
    assert session1 != session2

def test_autoscaling_policy_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AutoscalingPolicyServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_autoscaling_policy_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AutoscalingPolicyServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport])
def test_autoscaling_policy_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AutoscalingPolicyServiceGrpcTransport, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport])
def test_autoscaling_policy_service_transport_channel_mtls_with_adc(transport_class):
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

def test_autoscaling_policy_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    autoscaling_policy = 'whelk'
    expected = 'projects/{project}/locations/{location}/autoscalingPolicies/{autoscaling_policy}'.format(project=project, location=location, autoscaling_policy=autoscaling_policy)
    actual = AutoscalingPolicyServiceClient.autoscaling_policy_path(project, location, autoscaling_policy)
    assert expected == actual

def test_parse_autoscaling_policy_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'autoscaling_policy': 'nudibranch'}
    path = AutoscalingPolicyServiceClient.autoscaling_policy_path(**expected)
    actual = AutoscalingPolicyServiceClient.parse_autoscaling_policy_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AutoscalingPolicyServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = AutoscalingPolicyServiceClient.common_billing_account_path(**expected)
    actual = AutoscalingPolicyServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AutoscalingPolicyServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = AutoscalingPolicyServiceClient.common_folder_path(**expected)
    actual = AutoscalingPolicyServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AutoscalingPolicyServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'abalone'}
    path = AutoscalingPolicyServiceClient.common_organization_path(**expected)
    actual = AutoscalingPolicyServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = AutoscalingPolicyServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'clam'}
    path = AutoscalingPolicyServiceClient.common_project_path(**expected)
    actual = AutoscalingPolicyServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AutoscalingPolicyServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = AutoscalingPolicyServiceClient.common_location_path(**expected)
    actual = AutoscalingPolicyServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AutoscalingPolicyServiceTransport, '_prep_wrapped_messages') as prep:
        client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AutoscalingPolicyServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = AutoscalingPolicyServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/regions/sample2/clusters/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/regions/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/regions/sample2/clusters/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/regions/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/regions/sample2/clusters/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/regions/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/regions/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.CancelOperationRequest, dict])
def test_cancel_operation_rest(request_type):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/regions/sample2/operations/sample3'}
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
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/regions/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/regions/sample2/operations/sample3'}
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
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/regions/sample2/operations/sample3'}, request)
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/regions/sample2/operations/sample3'}
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
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/regions/sample2/operations'}, request)
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/regions/sample2/operations'}
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
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio'):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_field_headers():
    if False:
        return 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_set_iam_policy_from_dict():
    if False:
        i = 10
        return i + 15
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio'):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_get_iam_policy_from_dict():
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio'):
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_field_headers():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_test_iam_permissions_from_dict():
    if False:
        print('Hello World!')
    client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = AutoscalingPolicyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = AutoscalingPolicyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AutoscalingPolicyServiceClient, transports.AutoscalingPolicyServiceGrpcTransport), (AutoscalingPolicyServiceAsyncClient, transports.AutoscalingPolicyServiceGrpcAsyncIOTransport)])
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