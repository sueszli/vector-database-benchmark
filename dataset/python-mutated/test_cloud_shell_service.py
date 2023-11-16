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
from google.cloud.shell_v1.services.cloud_shell_service import CloudShellServiceAsyncClient, CloudShellServiceClient, transports
from google.cloud.shell_v1.types import cloudshell

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
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
    assert CloudShellServiceClient._get_default_mtls_endpoint(None) is None
    assert CloudShellServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudShellServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudShellServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudShellServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudShellServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudShellServiceClient, 'grpc'), (CloudShellServiceAsyncClient, 'grpc_asyncio'), (CloudShellServiceClient, 'rest')])
def test_cloud_shell_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudshell.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudshell.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudShellServiceGrpcTransport, 'grpc'), (transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudShellServiceRestTransport, 'rest')])
def test_cloud_shell_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CloudShellServiceClient, 'grpc'), (CloudShellServiceAsyncClient, 'grpc_asyncio'), (CloudShellServiceClient, 'rest')])
def test_cloud_shell_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudshell.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudshell.googleapis.com')

def test_cloud_shell_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = CloudShellServiceClient.get_transport_class()
    available_transports = [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceRestTransport]
    assert transport in available_transports
    transport = CloudShellServiceClient.get_transport_class('grpc')
    assert transport == transports.CloudShellServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudShellServiceClient, transports.CloudShellServiceGrpcTransport, 'grpc'), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudShellServiceClient, transports.CloudShellServiceRestTransport, 'rest')])
@mock.patch.object(CloudShellServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudShellServiceClient))
@mock.patch.object(CloudShellServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudShellServiceAsyncClient))
def test_cloud_shell_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(CloudShellServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudShellServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudShellServiceClient, transports.CloudShellServiceGrpcTransport, 'grpc', 'true'), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudShellServiceClient, transports.CloudShellServiceGrpcTransport, 'grpc', 'false'), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudShellServiceClient, transports.CloudShellServiceRestTransport, 'rest', 'true'), (CloudShellServiceClient, transports.CloudShellServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CloudShellServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudShellServiceClient))
@mock.patch.object(CloudShellServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudShellServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_shell_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudShellServiceClient, CloudShellServiceAsyncClient])
@mock.patch.object(CloudShellServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudShellServiceClient))
@mock.patch.object(CloudShellServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudShellServiceAsyncClient))
def test_cloud_shell_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudShellServiceClient, transports.CloudShellServiceGrpcTransport, 'grpc'), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudShellServiceClient, transports.CloudShellServiceRestTransport, 'rest')])
def test_cloud_shell_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudShellServiceClient, transports.CloudShellServiceGrpcTransport, 'grpc', grpc_helpers), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudShellServiceClient, transports.CloudShellServiceRestTransport, 'rest', None)])
def test_cloud_shell_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_shell_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.shell_v1.services.cloud_shell_service.transports.CloudShellServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudShellServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudShellServiceClient, transports.CloudShellServiceGrpcTransport, 'grpc', grpc_helpers), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_shell_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudshell.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudshell.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloudshell.GetEnvironmentRequest, dict])
def test_get_environment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        call.return_value = cloudshell.Environment(name='name_value', id='id_value', docker_image='docker_image_value', state=cloudshell.Environment.State.SUSPENDED, web_host='web_host_value', ssh_username='ssh_username_value', ssh_host='ssh_host_value', ssh_port=882, public_keys=['public_keys_value'])
        response = client.get_environment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.GetEnvironmentRequest()
    assert isinstance(response, cloudshell.Environment)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.docker_image == 'docker_image_value'
    assert response.state == cloudshell.Environment.State.SUSPENDED
    assert response.web_host == 'web_host_value'
    assert response.ssh_username == 'ssh_username_value'
    assert response.ssh_host == 'ssh_host_value'
    assert response.ssh_port == 882
    assert response.public_keys == ['public_keys_value']

def test_get_environment_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        client.get_environment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.GetEnvironmentRequest()

@pytest.mark.asyncio
async def test_get_environment_async(transport: str='grpc_asyncio', request_type=cloudshell.GetEnvironmentRequest):
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudshell.Environment(name='name_value', id='id_value', docker_image='docker_image_value', state=cloudshell.Environment.State.SUSPENDED, web_host='web_host_value', ssh_username='ssh_username_value', ssh_host='ssh_host_value', ssh_port=882, public_keys=['public_keys_value']))
        response = await client.get_environment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.GetEnvironmentRequest()
    assert isinstance(response, cloudshell.Environment)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.docker_image == 'docker_image_value'
    assert response.state == cloudshell.Environment.State.SUSPENDED
    assert response.web_host == 'web_host_value'
    assert response.ssh_username == 'ssh_username_value'
    assert response.ssh_host == 'ssh_host_value'
    assert response.ssh_port == 882
    assert response.public_keys == ['public_keys_value']

@pytest.mark.asyncio
async def test_get_environment_async_from_dict():
    await test_get_environment_async(request_type=dict)

def test_get_environment_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.GetEnvironmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        call.return_value = cloudshell.Environment()
        client.get_environment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_environment_field_headers_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.GetEnvironmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudshell.Environment())
        await client.get_environment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_environment_flattened():
    if False:
        print('Hello World!')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        call.return_value = cloudshell.Environment()
        client.get_environment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_environment_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_environment(cloudshell.GetEnvironmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_environment_flattened_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_environment), '__call__') as call:
        call.return_value = cloudshell.Environment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudshell.Environment())
        response = await client.get_environment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_environment_flattened_error_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_environment(cloudshell.GetEnvironmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudshell.StartEnvironmentRequest, dict])
def test_start_environment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_environment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_environment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.StartEnvironmentRequest()
    assert isinstance(response, future.Future)

def test_start_environment_empty_call():
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_environment), '__call__') as call:
        client.start_environment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.StartEnvironmentRequest()

@pytest.mark.asyncio
async def test_start_environment_async(transport: str='grpc_asyncio', request_type=cloudshell.StartEnvironmentRequest):
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_environment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_environment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.StartEnvironmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_environment_async_from_dict():
    await test_start_environment_async(request_type=dict)

def test_start_environment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.StartEnvironmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_environment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_environment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_environment_field_headers_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.StartEnvironmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_environment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_environment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [cloudshell.AuthorizeEnvironmentRequest, dict])
def test_authorize_environment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.authorize_environment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.authorize_environment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.AuthorizeEnvironmentRequest()
    assert isinstance(response, future.Future)

def test_authorize_environment_empty_call():
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.authorize_environment), '__call__') as call:
        client.authorize_environment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.AuthorizeEnvironmentRequest()

@pytest.mark.asyncio
async def test_authorize_environment_async(transport: str='grpc_asyncio', request_type=cloudshell.AuthorizeEnvironmentRequest):
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.authorize_environment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.authorize_environment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.AuthorizeEnvironmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_authorize_environment_async_from_dict():
    await test_authorize_environment_async(request_type=dict)

def test_authorize_environment_field_headers():
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.AuthorizeEnvironmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.authorize_environment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.authorize_environment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_authorize_environment_field_headers_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.AuthorizeEnvironmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.authorize_environment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.authorize_environment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [cloudshell.AddPublicKeyRequest, dict])
def test_add_public_key(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_public_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.add_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.AddPublicKeyRequest()
    assert isinstance(response, future.Future)

def test_add_public_key_empty_call():
    if False:
        print('Hello World!')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.add_public_key), '__call__') as call:
        client.add_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.AddPublicKeyRequest()

@pytest.mark.asyncio
async def test_add_public_key_async(transport: str='grpc_asyncio', request_type=cloudshell.AddPublicKeyRequest):
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.add_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.AddPublicKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_add_public_key_async_from_dict():
    await test_add_public_key_async(request_type=dict)

def test_add_public_key_field_headers():
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.AddPublicKeyRequest()
    request.environment = 'environment_value'
    with mock.patch.object(type(client.transport.add_public_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.add_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'environment=environment_value') in kw['metadata']

@pytest.mark.asyncio
async def test_add_public_key_field_headers_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.AddPublicKeyRequest()
    request.environment = 'environment_value'
    with mock.patch.object(type(client.transport.add_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.add_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'environment=environment_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [cloudshell.RemovePublicKeyRequest, dict])
def test_remove_public_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_public_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.remove_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.RemovePublicKeyRequest()
    assert isinstance(response, future.Future)

def test_remove_public_key_empty_call():
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.remove_public_key), '__call__') as call:
        client.remove_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.RemovePublicKeyRequest()

@pytest.mark.asyncio
async def test_remove_public_key_async(transport: str='grpc_asyncio', request_type=cloudshell.RemovePublicKeyRequest):
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.remove_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.remove_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudshell.RemovePublicKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_remove_public_key_async_from_dict():
    await test_remove_public_key_async(request_type=dict)

def test_remove_public_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.RemovePublicKeyRequest()
    request.environment = 'environment_value'
    with mock.patch.object(type(client.transport.remove_public_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.remove_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'environment=environment_value') in kw['metadata']

@pytest.mark.asyncio
async def test_remove_public_key_field_headers_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudshell.RemovePublicKeyRequest()
    request.environment = 'environment_value'
    with mock.patch.object(type(client.transport.remove_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.remove_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'environment=environment_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [cloudshell.GetEnvironmentRequest, dict])
def test_get_environment_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudshell.Environment(name='name_value', id='id_value', docker_image='docker_image_value', state=cloudshell.Environment.State.SUSPENDED, web_host='web_host_value', ssh_username='ssh_username_value', ssh_host='ssh_host_value', ssh_port=882, public_keys=['public_keys_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudshell.Environment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_environment(request)
    assert isinstance(response, cloudshell.Environment)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.docker_image == 'docker_image_value'
    assert response.state == cloudshell.Environment.State.SUSPENDED
    assert response.web_host == 'web_host_value'
    assert response.ssh_username == 'ssh_username_value'
    assert response.ssh_host == 'ssh_host_value'
    assert response.ssh_port == 882
    assert response.public_keys == ['public_keys_value']

def test_get_environment_rest_required_fields(request_type=cloudshell.GetEnvironmentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudShellServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_environment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_environment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloudshell.Environment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloudshell.Environment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_environment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_environment_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudShellServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_environment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_environment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudShellServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudShellServiceRestInterceptor())
    client = CloudShellServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudShellServiceRestInterceptor, 'post_get_environment') as post, mock.patch.object(transports.CloudShellServiceRestInterceptor, 'pre_get_environment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudshell.GetEnvironmentRequest.pb(cloudshell.GetEnvironmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloudshell.Environment.to_json(cloudshell.Environment())
        request = cloudshell.GetEnvironmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloudshell.Environment()
        client.get_environment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_environment_rest_bad_request(transport: str='rest', request_type=cloudshell.GetEnvironmentRequest):
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_environment(request)

def test_get_environment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudshell.Environment()
        sample_request = {'name': 'users/sample1/environments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudshell.Environment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_environment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=users/*/environments/*}' % client.transport._host, args[1])

def test_get_environment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_environment(cloudshell.GetEnvironmentRequest(), name='name_value')

def test_get_environment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudshell.StartEnvironmentRequest, dict])
def test_start_environment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_environment(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_environment_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudShellServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudShellServiceRestInterceptor())
    client = CloudShellServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudShellServiceRestInterceptor, 'post_start_environment') as post, mock.patch.object(transports.CloudShellServiceRestInterceptor, 'pre_start_environment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudshell.StartEnvironmentRequest.pb(cloudshell.StartEnvironmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloudshell.StartEnvironmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.start_environment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_environment_rest_bad_request(transport: str='rest', request_type=cloudshell.StartEnvironmentRequest):
    if False:
        print('Hello World!')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_environment(request)

def test_start_environment_rest_error():
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudshell.AuthorizeEnvironmentRequest, dict])
def test_authorize_environment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.authorize_environment(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_authorize_environment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudShellServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudShellServiceRestInterceptor())
    client = CloudShellServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudShellServiceRestInterceptor, 'post_authorize_environment') as post, mock.patch.object(transports.CloudShellServiceRestInterceptor, 'pre_authorize_environment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudshell.AuthorizeEnvironmentRequest.pb(cloudshell.AuthorizeEnvironmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloudshell.AuthorizeEnvironmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.authorize_environment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_authorize_environment_rest_bad_request(transport: str='rest', request_type=cloudshell.AuthorizeEnvironmentRequest):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.authorize_environment(request)

def test_authorize_environment_rest_error():
    if False:
        print('Hello World!')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudshell.AddPublicKeyRequest, dict])
def test_add_public_key_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'environment': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_public_key(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_public_key_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudShellServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudShellServiceRestInterceptor())
    client = CloudShellServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudShellServiceRestInterceptor, 'post_add_public_key') as post, mock.patch.object(transports.CloudShellServiceRestInterceptor, 'pre_add_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudshell.AddPublicKeyRequest.pb(cloudshell.AddPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloudshell.AddPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.add_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_public_key_rest_bad_request(transport: str='rest', request_type=cloudshell.AddPublicKeyRequest):
    if False:
        while True:
            i = 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'environment': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_public_key(request)

def test_add_public_key_rest_error():
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudshell.RemovePublicKeyRequest, dict])
def test_remove_public_key_rest(request_type):
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'environment': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_public_key(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_public_key_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudShellServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudShellServiceRestInterceptor())
    client = CloudShellServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudShellServiceRestInterceptor, 'post_remove_public_key') as post, mock.patch.object(transports.CloudShellServiceRestInterceptor, 'pre_remove_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudshell.RemovePublicKeyRequest.pb(cloudshell.RemovePublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloudshell.RemovePublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.remove_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_public_key_rest_bad_request(transport: str='rest', request_type=cloudshell.RemovePublicKeyRequest):
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'environment': 'users/sample1/environments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_public_key(request)

def test_remove_public_key_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudShellServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudShellServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudShellServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudShellServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudShellServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudShellServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudShellServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudShellServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudShellServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudShellServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.CloudShellServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudShellServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceGrpcAsyncIOTransport, transports.CloudShellServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = CloudShellServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudShellServiceGrpcTransport)

def test_cloud_shell_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudShellServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_shell_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.shell_v1.services.cloud_shell_service.transports.CloudShellServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudShellServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_environment', 'start_environment', 'authorize_environment', 'add_public_key', 'remove_public_key')
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

def test_cloud_shell_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.shell_v1.services.cloud_shell_service.transports.CloudShellServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudShellServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_shell_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.shell_v1.services.cloud_shell_service.transports.CloudShellServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudShellServiceTransport()
        adc.assert_called_once()

def test_cloud_shell_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudShellServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceGrpcAsyncIOTransport])
def test_cloud_shell_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceGrpcAsyncIOTransport, transports.CloudShellServiceRestTransport])
def test_cloud_shell_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudShellServiceGrpcTransport, grpc_helpers), (transports.CloudShellServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_shell_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudshell.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudshell.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceGrpcAsyncIOTransport])
def test_cloud_shell_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_cloud_shell_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudShellServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_cloud_shell_service_rest_lro_client():
    if False:
        print('Hello World!')
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_shell_service_host_no_port(transport_name):
    if False:
        return 10
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudshell.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudshell.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudshell.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_shell_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudshell.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudshell.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudshell.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_shell_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudShellServiceClient(credentials=creds1, transport=transport_name)
    client2 = CloudShellServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_environment._session
    session2 = client2.transport.get_environment._session
    assert session1 != session2
    session1 = client1.transport.start_environment._session
    session2 = client2.transport.start_environment._session
    assert session1 != session2
    session1 = client1.transport.authorize_environment._session
    session2 = client2.transport.authorize_environment._session
    assert session1 != session2
    session1 = client1.transport.add_public_key._session
    session2 = client2.transport.add_public_key._session
    assert session1 != session2
    session1 = client1.transport.remove_public_key._session
    session2 = client2.transport.remove_public_key._session
    assert session1 != session2

def test_cloud_shell_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudShellServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_shell_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudShellServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceGrpcAsyncIOTransport])
def test_cloud_shell_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CloudShellServiceGrpcTransport, transports.CloudShellServiceGrpcAsyncIOTransport])
def test_cloud_shell_service_transport_channel_mtls_with_adc(transport_class):
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

def test_cloud_shell_service_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_shell_service_grpc_lro_async_client():
    if False:
        return 10
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_environment_path():
    if False:
        for i in range(10):
            print('nop')
    user = 'squid'
    environment = 'clam'
    expected = 'users/{user}/environments/{environment}'.format(user=user, environment=environment)
    actual = CloudShellServiceClient.environment_path(user, environment)
    assert expected == actual

def test_parse_environment_path():
    if False:
        i = 10
        return i + 15
    expected = {'user': 'whelk', 'environment': 'octopus'}
    path = CloudShellServiceClient.environment_path(**expected)
    actual = CloudShellServiceClient.parse_environment_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudShellServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'nudibranch'}
    path = CloudShellServiceClient.common_billing_account_path(**expected)
    actual = CloudShellServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudShellServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'mussel'}
    path = CloudShellServiceClient.common_folder_path(**expected)
    actual = CloudShellServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudShellServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nautilus'}
    path = CloudShellServiceClient.common_organization_path(**expected)
    actual = CloudShellServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudShellServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone'}
    path = CloudShellServiceClient.common_project_path(**expected)
    actual = CloudShellServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudShellServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = CloudShellServiceClient.common_location_path(**expected)
    actual = CloudShellServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudShellServiceTransport, '_prep_wrapped_messages') as prep:
        client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudShellServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudShellServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudShellServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CloudShellServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudShellServiceClient, transports.CloudShellServiceGrpcTransport), (CloudShellServiceAsyncClient, transports.CloudShellServiceGrpcAsyncIOTransport)])
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