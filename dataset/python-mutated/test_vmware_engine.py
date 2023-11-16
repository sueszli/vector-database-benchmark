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
from google.cloud.location import locations_pb2
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.vmwareengine_v1.services.vmware_engine import VmwareEngineAsyncClient, VmwareEngineClient, pagers, transports
from google.cloud.vmwareengine_v1.types import vmwareengine, vmwareengine_resources

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
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
    assert VmwareEngineClient._get_default_mtls_endpoint(None) is None
    assert VmwareEngineClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert VmwareEngineClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert VmwareEngineClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert VmwareEngineClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert VmwareEngineClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(VmwareEngineClient, 'grpc'), (VmwareEngineAsyncClient, 'grpc_asyncio'), (VmwareEngineClient, 'rest')])
def test_vmware_engine_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('vmwareengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmwareengine.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.VmwareEngineGrpcTransport, 'grpc'), (transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.VmwareEngineRestTransport, 'rest')])
def test_vmware_engine_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(VmwareEngineClient, 'grpc'), (VmwareEngineAsyncClient, 'grpc_asyncio'), (VmwareEngineClient, 'rest')])
def test_vmware_engine_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('vmwareengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmwareengine.googleapis.com')

def test_vmware_engine_client_get_transport_class():
    if False:
        return 10
    transport = VmwareEngineClient.get_transport_class()
    available_transports = [transports.VmwareEngineGrpcTransport, transports.VmwareEngineRestTransport]
    assert transport in available_transports
    transport = VmwareEngineClient.get_transport_class('grpc')
    assert transport == transports.VmwareEngineGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VmwareEngineClient, transports.VmwareEngineGrpcTransport, 'grpc'), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio'), (VmwareEngineClient, transports.VmwareEngineRestTransport, 'rest')])
@mock.patch.object(VmwareEngineClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmwareEngineClient))
@mock.patch.object(VmwareEngineAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmwareEngineAsyncClient))
def test_vmware_engine_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(VmwareEngineClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(VmwareEngineClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(VmwareEngineClient, transports.VmwareEngineGrpcTransport, 'grpc', 'true'), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (VmwareEngineClient, transports.VmwareEngineGrpcTransport, 'grpc', 'false'), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (VmwareEngineClient, transports.VmwareEngineRestTransport, 'rest', 'true'), (VmwareEngineClient, transports.VmwareEngineRestTransport, 'rest', 'false')])
@mock.patch.object(VmwareEngineClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmwareEngineClient))
@mock.patch.object(VmwareEngineAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmwareEngineAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_vmware_engine_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [VmwareEngineClient, VmwareEngineAsyncClient])
@mock.patch.object(VmwareEngineClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmwareEngineClient))
@mock.patch.object(VmwareEngineAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VmwareEngineAsyncClient))
def test_vmware_engine_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VmwareEngineClient, transports.VmwareEngineGrpcTransport, 'grpc'), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio'), (VmwareEngineClient, transports.VmwareEngineRestTransport, 'rest')])
def test_vmware_engine_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VmwareEngineClient, transports.VmwareEngineGrpcTransport, 'grpc', grpc_helpers), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (VmwareEngineClient, transports.VmwareEngineRestTransport, 'rest', None)])
def test_vmware_engine_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_vmware_engine_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.vmwareengine_v1.services.vmware_engine.transports.VmwareEngineGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = VmwareEngineClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VmwareEngineClient, transports.VmwareEngineGrpcTransport, 'grpc', grpc_helpers), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_vmware_engine_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
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
        create_channel.assert_called_with('vmwareengine.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='vmwareengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [vmwareengine.ListPrivateCloudsRequest, dict])
def test_list_private_clouds(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateCloudsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_private_clouds(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateCloudsRequest()
    assert isinstance(response, pagers.ListPrivateCloudsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_private_clouds_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        client.list_private_clouds()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateCloudsRequest()

@pytest.mark.asyncio
async def test_list_private_clouds_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListPrivateCloudsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateCloudsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_private_clouds(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateCloudsRequest()
    assert isinstance(response, pagers.ListPrivateCloudsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_private_clouds_async_from_dict():
    await test_list_private_clouds_async(request_type=dict)

def test_list_private_clouds_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListPrivateCloudsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateCloudsResponse()
        client.list_private_clouds(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_private_clouds_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListPrivateCloudsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateCloudsResponse())
        await client.list_private_clouds(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_private_clouds_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateCloudsResponse()
        client.list_private_clouds(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_private_clouds_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_private_clouds(vmwareengine.ListPrivateCloudsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_private_clouds_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateCloudsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateCloudsResponse())
        response = await client.list_private_clouds(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_private_clouds_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_private_clouds(vmwareengine.ListPrivateCloudsRequest(), parent='parent_value')

def test_list_private_clouds_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.side_effect = (vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()], next_page_token='abc'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[], next_page_token='def'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud()], next_page_token='ghi'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_private_clouds(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.PrivateCloud) for i in results))

def test_list_private_clouds_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__') as call:
        call.side_effect = (vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()], next_page_token='abc'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[], next_page_token='def'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud()], next_page_token='ghi'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()]), RuntimeError)
        pages = list(client.list_private_clouds(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_private_clouds_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()], next_page_token='abc'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[], next_page_token='def'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud()], next_page_token='ghi'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()]), RuntimeError)
        async_pager = await client.list_private_clouds(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.PrivateCloud) for i in responses))

@pytest.mark.asyncio
async def test_list_private_clouds_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_private_clouds), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()], next_page_token='abc'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[], next_page_token='def'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud()], next_page_token='ghi'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_private_clouds(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetPrivateCloudRequest, dict])
def test_get_private_cloud(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateCloud(name='name_value', state=vmwareengine_resources.PrivateCloud.State.ACTIVE, description='description_value', uid='uid_value', type_=vmwareengine_resources.PrivateCloud.Type.TIME_LIMITED)
        response = client.get_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetPrivateCloudRequest()
    assert isinstance(response, vmwareengine_resources.PrivateCloud)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.PrivateCloud.State.ACTIVE
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.type_ == vmwareengine_resources.PrivateCloud.Type.TIME_LIMITED

def test_get_private_cloud_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        client.get_private_cloud()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetPrivateCloudRequest()

@pytest.mark.asyncio
async def test_get_private_cloud_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetPrivateCloudRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.PrivateCloud(name='name_value', state=vmwareengine_resources.PrivateCloud.State.ACTIVE, description='description_value', uid='uid_value', type_=vmwareengine_resources.PrivateCloud.Type.TIME_LIMITED))
        response = await client.get_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetPrivateCloudRequest()
    assert isinstance(response, vmwareengine_resources.PrivateCloud)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.PrivateCloud.State.ACTIVE
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.type_ == vmwareengine_resources.PrivateCloud.Type.TIME_LIMITED

@pytest.mark.asyncio
async def test_get_private_cloud_async_from_dict():
    await test_get_private_cloud_async(request_type=dict)

def test_get_private_cloud_field_headers():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetPrivateCloudRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateCloud()
        client.get_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_private_cloud_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetPrivateCloudRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.PrivateCloud())
        await client.get_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_private_cloud_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateCloud()
        client.get_private_cloud(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_private_cloud_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_private_cloud(vmwareengine.GetPrivateCloudRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_private_cloud_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_private_cloud), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateCloud()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.PrivateCloud())
        response = await client.get_private_cloud(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_private_cloud_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_private_cloud(vmwareengine.GetPrivateCloudRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.CreatePrivateCloudRequest, dict])
def test_create_private_cloud(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreatePrivateCloudRequest()
    assert isinstance(response, future.Future)

def test_create_private_cloud_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        client.create_private_cloud()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreatePrivateCloudRequest()

@pytest.mark.asyncio
async def test_create_private_cloud_async(transport: str='grpc_asyncio', request_type=vmwareengine.CreatePrivateCloudRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreatePrivateCloudRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_private_cloud_async_from_dict():
    await test_create_private_cloud_async(request_type=dict)

def test_create_private_cloud_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreatePrivateCloudRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_private_cloud_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreatePrivateCloudRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_private_cloud_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_private_cloud(parent='parent_value', private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), private_cloud_id='private_cloud_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].private_cloud
        mock_val = vmwareengine_resources.PrivateCloud(name='name_value')
        assert arg == mock_val
        arg = args[0].private_cloud_id
        mock_val = 'private_cloud_id_value'
        assert arg == mock_val

def test_create_private_cloud_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_private_cloud(vmwareengine.CreatePrivateCloudRequest(), parent='parent_value', private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), private_cloud_id='private_cloud_id_value')

@pytest.mark.asyncio
async def test_create_private_cloud_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_private_cloud(parent='parent_value', private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), private_cloud_id='private_cloud_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].private_cloud
        mock_val = vmwareengine_resources.PrivateCloud(name='name_value')
        assert arg == mock_val
        arg = args[0].private_cloud_id
        mock_val = 'private_cloud_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_private_cloud_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_private_cloud(vmwareengine.CreatePrivateCloudRequest(), parent='parent_value', private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), private_cloud_id='private_cloud_id_value')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdatePrivateCloudRequest, dict])
def test_update_private_cloud(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdatePrivateCloudRequest()
    assert isinstance(response, future.Future)

def test_update_private_cloud_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        client.update_private_cloud()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdatePrivateCloudRequest()

@pytest.mark.asyncio
async def test_update_private_cloud_async(transport: str='grpc_asyncio', request_type=vmwareengine.UpdatePrivateCloudRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdatePrivateCloudRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_private_cloud_async_from_dict():
    await test_update_private_cloud_async(request_type=dict)

def test_update_private_cloud_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdatePrivateCloudRequest()
    request.private_cloud.name = 'name_value'
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_private_cloud_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdatePrivateCloudRequest()
    request.private_cloud.name = 'name_value'
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud.name=name_value') in kw['metadata']

def test_update_private_cloud_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_private_cloud(private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = vmwareengine_resources.PrivateCloud(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_private_cloud_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_private_cloud(vmwareengine.UpdatePrivateCloudRequest(), private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_private_cloud_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_private_cloud(private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = vmwareengine_resources.PrivateCloud(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_private_cloud_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_private_cloud(vmwareengine.UpdatePrivateCloudRequest(), private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmwareengine.DeletePrivateCloudRequest, dict])
def test_delete_private_cloud(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeletePrivateCloudRequest()
    assert isinstance(response, future.Future)

def test_delete_private_cloud_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        client.delete_private_cloud()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeletePrivateCloudRequest()

@pytest.mark.asyncio
async def test_delete_private_cloud_async(transport: str='grpc_asyncio', request_type=vmwareengine.DeletePrivateCloudRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeletePrivateCloudRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_private_cloud_async_from_dict():
    await test_delete_private_cloud_async(request_type=dict)

def test_delete_private_cloud_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeletePrivateCloudRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_private_cloud_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeletePrivateCloudRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_private_cloud_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_private_cloud(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_private_cloud_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_private_cloud(vmwareengine.DeletePrivateCloudRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_private_cloud_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_private_cloud(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_private_cloud_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_private_cloud(vmwareengine.DeletePrivateCloudRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.UndeletePrivateCloudRequest, dict])
def test_undelete_private_cloud(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.undelete_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UndeletePrivateCloudRequest()
    assert isinstance(response, future.Future)

def test_undelete_private_cloud_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        client.undelete_private_cloud()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UndeletePrivateCloudRequest()

@pytest.mark.asyncio
async def test_undelete_private_cloud_async(transport: str='grpc_asyncio', request_type=vmwareengine.UndeletePrivateCloudRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undelete_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UndeletePrivateCloudRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_undelete_private_cloud_async_from_dict():
    await test_undelete_private_cloud_async(request_type=dict)

def test_undelete_private_cloud_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UndeletePrivateCloudRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undelete_private_cloud(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_undelete_private_cloud_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UndeletePrivateCloudRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.undelete_private_cloud(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_undelete_private_cloud_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undelete_private_cloud(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_undelete_private_cloud_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.undelete_private_cloud(vmwareengine.UndeletePrivateCloudRequest(), name='name_value')

@pytest.mark.asyncio
async def test_undelete_private_cloud_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undelete_private_cloud), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undelete_private_cloud(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_undelete_private_cloud_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.undelete_private_cloud(vmwareengine.UndeletePrivateCloudRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListClustersRequest, dict])
def test_list_clusters(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = vmwareengine.ListClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListClustersRequest()
    assert isinstance(response, pagers.ListClustersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_clusters_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        client.list_clusters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListClustersRequest()

@pytest.mark.asyncio
async def test_list_clusters_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListClustersRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListClustersRequest()
    assert isinstance(response, pagers.ListClustersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_clusters_async_from_dict():
    await test_list_clusters_async(request_type=dict)

def test_list_clusters_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = vmwareengine.ListClustersResponse()
        client.list_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_clusters_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListClustersResponse())
        await client.list_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_clusters_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = vmwareengine.ListClustersResponse()
        client.list_clusters(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_clusters_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_clusters(vmwareengine.ListClustersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_clusters_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = vmwareengine.ListClustersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListClustersResponse())
        response = await client.list_clusters(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_clusters_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_clusters(vmwareengine.ListClustersRequest(), parent='parent_value')

def test_list_clusters_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.side_effect = (vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()], next_page_token='abc'), vmwareengine.ListClustersResponse(clusters=[], next_page_token='def'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster()], next_page_token='ghi'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_clusters(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.Cluster) for i in results))

def test_list_clusters_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.side_effect = (vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()], next_page_token='abc'), vmwareengine.ListClustersResponse(clusters=[], next_page_token='def'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster()], next_page_token='ghi'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()]), RuntimeError)
        pages = list(client.list_clusters(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_clusters_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()], next_page_token='abc'), vmwareengine.ListClustersResponse(clusters=[], next_page_token='def'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster()], next_page_token='ghi'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()]), RuntimeError)
        async_pager = await client.list_clusters(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.Cluster) for i in responses))

@pytest.mark.asyncio
async def test_list_clusters_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()], next_page_token='abc'), vmwareengine.ListClustersResponse(clusters=[], next_page_token='def'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster()], next_page_token='ghi'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_clusters(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetClusterRequest, dict])
def test_get_cluster(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = vmwareengine_resources.Cluster(name='name_value', state=vmwareengine_resources.Cluster.State.ACTIVE, management=True, uid='uid_value')
        response = client.get_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetClusterRequest()
    assert isinstance(response, vmwareengine_resources.Cluster)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.Cluster.State.ACTIVE
    assert response.management is True
    assert response.uid == 'uid_value'

def test_get_cluster_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        client.get_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetClusterRequest()

@pytest.mark.asyncio
async def test_get_cluster_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetClusterRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Cluster(name='name_value', state=vmwareengine_resources.Cluster.State.ACTIVE, management=True, uid='uid_value'))
        response = await client.get_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetClusterRequest()
    assert isinstance(response, vmwareengine_resources.Cluster)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.Cluster.State.ACTIVE
    assert response.management is True
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_cluster_async_from_dict():
    await test_get_cluster_async(request_type=dict)

def test_get_cluster_field_headers():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = vmwareengine_resources.Cluster()
        client.get_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_cluster_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Cluster())
        await client.get_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_cluster_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = vmwareengine_resources.Cluster()
        client.get_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_cluster_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_cluster(vmwareengine.GetClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_cluster_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = vmwareengine_resources.Cluster()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Cluster())
        response = await client.get_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_cluster_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_cluster(vmwareengine.GetClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.CreateClusterRequest, dict])
def test_create_cluster(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateClusterRequest()
    assert isinstance(response, future.Future)

def test_create_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        client.create_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateClusterRequest()

@pytest.mark.asyncio
async def test_create_cluster_async(transport: str='grpc_asyncio', request_type=vmwareengine.CreateClusterRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_cluster_async_from_dict():
    await test_create_cluster_async(request_type=dict)

def test_create_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_cluster_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_cluster_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cluster(parent='parent_value', cluster=vmwareengine_resources.Cluster(name='name_value'), cluster_id='cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cluster
        mock_val = vmwareengine_resources.Cluster(name='name_value')
        assert arg == mock_val
        arg = args[0].cluster_id
        mock_val = 'cluster_id_value'
        assert arg == mock_val

def test_create_cluster_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_cluster(vmwareengine.CreateClusterRequest(), parent='parent_value', cluster=vmwareengine_resources.Cluster(name='name_value'), cluster_id='cluster_id_value')

@pytest.mark.asyncio
async def test_create_cluster_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cluster(parent='parent_value', cluster=vmwareengine_resources.Cluster(name='name_value'), cluster_id='cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cluster
        mock_val = vmwareengine_resources.Cluster(name='name_value')
        assert arg == mock_val
        arg = args[0].cluster_id
        mock_val = 'cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_cluster_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_cluster(vmwareengine.CreateClusterRequest(), parent='parent_value', cluster=vmwareengine_resources.Cluster(name='name_value'), cluster_id='cluster_id_value')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateClusterRequest, dict])
def test_update_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateClusterRequest()
    assert isinstance(response, future.Future)

def test_update_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        client.update_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateClusterRequest()

@pytest.mark.asyncio
async def test_update_cluster_async(transport: str='grpc_asyncio', request_type=vmwareengine.UpdateClusterRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_cluster_async_from_dict():
    await test_update_cluster_async(request_type=dict)

def test_update_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateClusterRequest()
    request.cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'cluster.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_cluster_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateClusterRequest()
    request.cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'cluster.name=name_value') in kw['metadata']

def test_update_cluster_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_cluster(cluster=vmwareengine_resources.Cluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].cluster
        mock_val = vmwareengine_resources.Cluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_cluster_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_cluster(vmwareengine.UpdateClusterRequest(), cluster=vmwareengine_resources.Cluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_cluster_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_cluster(cluster=vmwareengine_resources.Cluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].cluster
        mock_val = vmwareengine_resources.Cluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_cluster_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_cluster(vmwareengine.UpdateClusterRequest(), cluster=vmwareengine_resources.Cluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmwareengine.DeleteClusterRequest, dict])
def test_delete_cluster(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteClusterRequest()
    assert isinstance(response, future.Future)

def test_delete_cluster_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        client.delete_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteClusterRequest()

@pytest.mark.asyncio
async def test_delete_cluster_async(transport: str='grpc_asyncio', request_type=vmwareengine.DeleteClusterRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_cluster_async_from_dict():
    await test_delete_cluster_async(request_type=dict)

def test_delete_cluster_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeleteClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_cluster_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeleteClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_cluster_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_cluster_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_cluster(vmwareengine.DeleteClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_cluster_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_cluster_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_cluster(vmwareengine.DeleteClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListSubnetsRequest, dict])
def test_list_subnets(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = vmwareengine.ListSubnetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_subnets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListSubnetsRequest()
    assert isinstance(response, pagers.ListSubnetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_subnets_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        client.list_subnets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListSubnetsRequest()

@pytest.mark.asyncio
async def test_list_subnets_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListSubnetsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListSubnetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_subnets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListSubnetsRequest()
    assert isinstance(response, pagers.ListSubnetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_subnets_async_from_dict():
    await test_list_subnets_async(request_type=dict)

def test_list_subnets_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListSubnetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = vmwareengine.ListSubnetsResponse()
        client.list_subnets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_subnets_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListSubnetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListSubnetsResponse())
        await client.list_subnets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_subnets_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = vmwareengine.ListSubnetsResponse()
        client.list_subnets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_subnets_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_subnets(vmwareengine.ListSubnetsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_subnets_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = vmwareengine.ListSubnetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListSubnetsResponse())
        response = await client.list_subnets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_subnets_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_subnets(vmwareengine.ListSubnetsRequest(), parent='parent_value')

def test_list_subnets_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.side_effect = (vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()], next_page_token='abc'), vmwareengine.ListSubnetsResponse(subnets=[], next_page_token='def'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet()], next_page_token='ghi'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_subnets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.Subnet) for i in results))

def test_list_subnets_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.side_effect = (vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()], next_page_token='abc'), vmwareengine.ListSubnetsResponse(subnets=[], next_page_token='def'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet()], next_page_token='ghi'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()]), RuntimeError)
        pages = list(client.list_subnets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_subnets_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_subnets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()], next_page_token='abc'), vmwareengine.ListSubnetsResponse(subnets=[], next_page_token='def'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet()], next_page_token='ghi'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()]), RuntimeError)
        async_pager = await client.list_subnets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.Subnet) for i in responses))

@pytest.mark.asyncio
async def test_list_subnets_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_subnets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()], next_page_token='abc'), vmwareengine.ListSubnetsResponse(subnets=[], next_page_token='def'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet()], next_page_token='ghi'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_subnets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetSubnetRequest, dict])
def test_get_subnet(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = vmwareengine_resources.Subnet(name='name_value', ip_cidr_range='ip_cidr_range_value', gateway_ip='gateway_ip_value', type_='type__value', state=vmwareengine_resources.Subnet.State.ACTIVE)
        response = client.get_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetSubnetRequest()
    assert isinstance(response, vmwareengine_resources.Subnet)
    assert response.name == 'name_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.gateway_ip == 'gateway_ip_value'
    assert response.type_ == 'type__value'
    assert response.state == vmwareengine_resources.Subnet.State.ACTIVE

def test_get_subnet_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        client.get_subnet()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetSubnetRequest()

@pytest.mark.asyncio
async def test_get_subnet_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetSubnetRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Subnet(name='name_value', ip_cidr_range='ip_cidr_range_value', gateway_ip='gateway_ip_value', type_='type__value', state=vmwareengine_resources.Subnet.State.ACTIVE))
        response = await client.get_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetSubnetRequest()
    assert isinstance(response, vmwareengine_resources.Subnet)
    assert response.name == 'name_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.gateway_ip == 'gateway_ip_value'
    assert response.type_ == 'type__value'
    assert response.state == vmwareengine_resources.Subnet.State.ACTIVE

@pytest.mark.asyncio
async def test_get_subnet_async_from_dict():
    await test_get_subnet_async(request_type=dict)

def test_get_subnet_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetSubnetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = vmwareengine_resources.Subnet()
        client.get_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_subnet_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetSubnetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Subnet())
        await client.get_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_subnet_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = vmwareengine_resources.Subnet()
        client.get_subnet(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_subnet_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_subnet(vmwareengine.GetSubnetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_subnet_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = vmwareengine_resources.Subnet()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Subnet())
        response = await client.get_subnet(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_subnet_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_subnet(vmwareengine.GetSubnetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateSubnetRequest, dict])
def test_update_subnet(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateSubnetRequest()
    assert isinstance(response, future.Future)

def test_update_subnet_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        client.update_subnet()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateSubnetRequest()

@pytest.mark.asyncio
async def test_update_subnet_async(transport: str='grpc_asyncio', request_type=vmwareengine.UpdateSubnetRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateSubnetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_subnet_async_from_dict():
    await test_update_subnet_async(request_type=dict)

def test_update_subnet_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateSubnetRequest()
    request.subnet.name = 'name_value'
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'subnet.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_subnet_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateSubnetRequest()
    request.subnet.name = 'name_value'
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'subnet.name=name_value') in kw['metadata']

def test_update_subnet_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_subnet(subnet=vmwareengine_resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].subnet
        mock_val = vmwareengine_resources.Subnet(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_subnet_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_subnet(vmwareengine.UpdateSubnetRequest(), subnet=vmwareengine_resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_subnet_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_subnet(subnet=vmwareengine_resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].subnet
        mock_val = vmwareengine_resources.Subnet(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_subnet_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_subnet(vmwareengine.UpdateSubnetRequest(), subnet=vmwareengine_resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmwareengine.ListNodeTypesRequest, dict])
def test_list_node_types(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.return_value = vmwareengine.ListNodeTypesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_node_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListNodeTypesRequest()
    assert isinstance(response, pagers.ListNodeTypesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_node_types_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        client.list_node_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListNodeTypesRequest()

@pytest.mark.asyncio
async def test_list_node_types_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListNodeTypesRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListNodeTypesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_node_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListNodeTypesRequest()
    assert isinstance(response, pagers.ListNodeTypesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_node_types_async_from_dict():
    await test_list_node_types_async(request_type=dict)

def test_list_node_types_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListNodeTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.return_value = vmwareengine.ListNodeTypesResponse()
        client.list_node_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_node_types_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListNodeTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListNodeTypesResponse())
        await client.list_node_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_node_types_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.return_value = vmwareengine.ListNodeTypesResponse()
        client.list_node_types(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_node_types_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_node_types(vmwareengine.ListNodeTypesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_node_types_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.return_value = vmwareengine.ListNodeTypesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListNodeTypesResponse())
        response = await client.list_node_types(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_node_types_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_node_types(vmwareengine.ListNodeTypesRequest(), parent='parent_value')

def test_list_node_types_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.side_effect = (vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()], next_page_token='abc'), vmwareengine.ListNodeTypesResponse(node_types=[], next_page_token='def'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType()], next_page_token='ghi'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_node_types(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.NodeType) for i in results))

def test_list_node_types_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_node_types), '__call__') as call:
        call.side_effect = (vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()], next_page_token='abc'), vmwareengine.ListNodeTypesResponse(node_types=[], next_page_token='def'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType()], next_page_token='ghi'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()]), RuntimeError)
        pages = list(client.list_node_types(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_node_types_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_node_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()], next_page_token='abc'), vmwareengine.ListNodeTypesResponse(node_types=[], next_page_token='def'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType()], next_page_token='ghi'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()]), RuntimeError)
        async_pager = await client.list_node_types(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.NodeType) for i in responses))

@pytest.mark.asyncio
async def test_list_node_types_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_node_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()], next_page_token='abc'), vmwareengine.ListNodeTypesResponse(node_types=[], next_page_token='def'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType()], next_page_token='ghi'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_node_types(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetNodeTypeRequest, dict])
def test_get_node_type(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        call.return_value = vmwareengine_resources.NodeType(name='name_value', node_type_id='node_type_id_value', display_name='display_name_value', virtual_cpu_count=1846, total_core_count=1716, memory_gb=961, disk_size_gb=1261, available_custom_core_counts=[2974])
        response = client.get_node_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetNodeTypeRequest()
    assert isinstance(response, vmwareengine_resources.NodeType)
    assert response.name == 'name_value'
    assert response.node_type_id == 'node_type_id_value'
    assert response.display_name == 'display_name_value'
    assert response.virtual_cpu_count == 1846
    assert response.total_core_count == 1716
    assert response.memory_gb == 961
    assert response.disk_size_gb == 1261
    assert response.available_custom_core_counts == [2974]

def test_get_node_type_empty_call():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        client.get_node_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetNodeTypeRequest()

@pytest.mark.asyncio
async def test_get_node_type_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetNodeTypeRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.NodeType(name='name_value', node_type_id='node_type_id_value', display_name='display_name_value', virtual_cpu_count=1846, total_core_count=1716, memory_gb=961, disk_size_gb=1261, available_custom_core_counts=[2974]))
        response = await client.get_node_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetNodeTypeRequest()
    assert isinstance(response, vmwareengine_resources.NodeType)
    assert response.name == 'name_value'
    assert response.node_type_id == 'node_type_id_value'
    assert response.display_name == 'display_name_value'
    assert response.virtual_cpu_count == 1846
    assert response.total_core_count == 1716
    assert response.memory_gb == 961
    assert response.disk_size_gb == 1261
    assert response.available_custom_core_counts == [2974]

@pytest.mark.asyncio
async def test_get_node_type_async_from_dict():
    await test_get_node_type_async(request_type=dict)

def test_get_node_type_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetNodeTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        call.return_value = vmwareengine_resources.NodeType()
        client.get_node_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_node_type_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetNodeTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.NodeType())
        await client.get_node_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_node_type_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        call.return_value = vmwareengine_resources.NodeType()
        client.get_node_type(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_node_type_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_node_type(vmwareengine.GetNodeTypeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_node_type_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_node_type), '__call__') as call:
        call.return_value = vmwareengine_resources.NodeType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.NodeType())
        response = await client.get_node_type(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_node_type_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_node_type(vmwareengine.GetNodeTypeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ShowNsxCredentialsRequest, dict])
def test_show_nsx_credentials(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials(username='username_value', password='password_value')
        response = client.show_nsx_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ShowNsxCredentialsRequest()
    assert isinstance(response, vmwareengine_resources.Credentials)
    assert response.username == 'username_value'
    assert response.password == 'password_value'

def test_show_nsx_credentials_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        client.show_nsx_credentials()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ShowNsxCredentialsRequest()

@pytest.mark.asyncio
async def test_show_nsx_credentials_async(transport: str='grpc_asyncio', request_type=vmwareengine.ShowNsxCredentialsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Credentials(username='username_value', password='password_value'))
        response = await client.show_nsx_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ShowNsxCredentialsRequest()
    assert isinstance(response, vmwareengine_resources.Credentials)
    assert response.username == 'username_value'
    assert response.password == 'password_value'

@pytest.mark.asyncio
async def test_show_nsx_credentials_async_from_dict():
    await test_show_nsx_credentials_async(request_type=dict)

def test_show_nsx_credentials_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ShowNsxCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials()
        client.show_nsx_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

@pytest.mark.asyncio
async def test_show_nsx_credentials_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ShowNsxCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Credentials())
        await client.show_nsx_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

def test_show_nsx_credentials_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials()
        client.show_nsx_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

def test_show_nsx_credentials_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.show_nsx_credentials(vmwareengine.ShowNsxCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.asyncio
async def test_show_nsx_credentials_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.show_nsx_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Credentials())
        response = await client.show_nsx_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_show_nsx_credentials_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.show_nsx_credentials(vmwareengine.ShowNsxCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ShowVcenterCredentialsRequest, dict])
def test_show_vcenter_credentials(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials(username='username_value', password='password_value')
        response = client.show_vcenter_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ShowVcenterCredentialsRequest()
    assert isinstance(response, vmwareengine_resources.Credentials)
    assert response.username == 'username_value'
    assert response.password == 'password_value'

def test_show_vcenter_credentials_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        client.show_vcenter_credentials()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ShowVcenterCredentialsRequest()

@pytest.mark.asyncio
async def test_show_vcenter_credentials_async(transport: str='grpc_asyncio', request_type=vmwareengine.ShowVcenterCredentialsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Credentials(username='username_value', password='password_value'))
        response = await client.show_vcenter_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ShowVcenterCredentialsRequest()
    assert isinstance(response, vmwareengine_resources.Credentials)
    assert response.username == 'username_value'
    assert response.password == 'password_value'

@pytest.mark.asyncio
async def test_show_vcenter_credentials_async_from_dict():
    await test_show_vcenter_credentials_async(request_type=dict)

def test_show_vcenter_credentials_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ShowVcenterCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials()
        client.show_vcenter_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

@pytest.mark.asyncio
async def test_show_vcenter_credentials_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ShowVcenterCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Credentials())
        await client.show_vcenter_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

def test_show_vcenter_credentials_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials()
        client.show_vcenter_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

def test_show_vcenter_credentials_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.show_vcenter_credentials(vmwareengine.ShowVcenterCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.asyncio
async def test_show_vcenter_credentials_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.show_vcenter_credentials), '__call__') as call:
        call.return_value = vmwareengine_resources.Credentials()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.Credentials())
        response = await client.show_vcenter_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_show_vcenter_credentials_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.show_vcenter_credentials(vmwareengine.ShowVcenterCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ResetNsxCredentialsRequest, dict])
def test_reset_nsx_credentials(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reset_nsx_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ResetNsxCredentialsRequest()
    assert isinstance(response, future.Future)

def test_reset_nsx_credentials_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        client.reset_nsx_credentials()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ResetNsxCredentialsRequest()

@pytest.mark.asyncio
async def test_reset_nsx_credentials_async(transport: str='grpc_asyncio', request_type=vmwareengine.ResetNsxCredentialsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_nsx_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ResetNsxCredentialsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reset_nsx_credentials_async_from_dict():
    await test_reset_nsx_credentials_async(request_type=dict)

def test_reset_nsx_credentials_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ResetNsxCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_nsx_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reset_nsx_credentials_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ResetNsxCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reset_nsx_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

def test_reset_nsx_credentials_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_nsx_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

def test_reset_nsx_credentials_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reset_nsx_credentials(vmwareengine.ResetNsxCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.asyncio
async def test_reset_nsx_credentials_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_nsx_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_nsx_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reset_nsx_credentials_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reset_nsx_credentials(vmwareengine.ResetNsxCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ResetVcenterCredentialsRequest, dict])
def test_reset_vcenter_credentials(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reset_vcenter_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ResetVcenterCredentialsRequest()
    assert isinstance(response, future.Future)

def test_reset_vcenter_credentials_empty_call():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        client.reset_vcenter_credentials()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ResetVcenterCredentialsRequest()

@pytest.mark.asyncio
async def test_reset_vcenter_credentials_async(transport: str='grpc_asyncio', request_type=vmwareengine.ResetVcenterCredentialsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_vcenter_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ResetVcenterCredentialsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reset_vcenter_credentials_async_from_dict():
    await test_reset_vcenter_credentials_async(request_type=dict)

def test_reset_vcenter_credentials_field_headers():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ResetVcenterCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_vcenter_credentials(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reset_vcenter_credentials_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ResetVcenterCredentialsRequest()
    request.private_cloud = 'private_cloud_value'
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reset_vcenter_credentials(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_cloud=private_cloud_value') in kw['metadata']

def test_reset_vcenter_credentials_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_vcenter_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

def test_reset_vcenter_credentials_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reset_vcenter_credentials(vmwareengine.ResetVcenterCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.asyncio
async def test_reset_vcenter_credentials_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_vcenter_credentials), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_vcenter_credentials(private_cloud='private_cloud_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_cloud
        mock_val = 'private_cloud_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reset_vcenter_credentials_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reset_vcenter_credentials(vmwareengine.ResetVcenterCredentialsRequest(), private_cloud='private_cloud_value')

@pytest.mark.parametrize('request_type', [vmwareengine.CreateHcxActivationKeyRequest, dict])
def test_create_hcx_activation_key(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_hcx_activation_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateHcxActivationKeyRequest()
    assert isinstance(response, future.Future)

def test_create_hcx_activation_key_empty_call():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        client.create_hcx_activation_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateHcxActivationKeyRequest()

@pytest.mark.asyncio
async def test_create_hcx_activation_key_async(transport: str='grpc_asyncio', request_type=vmwareengine.CreateHcxActivationKeyRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_hcx_activation_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateHcxActivationKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_hcx_activation_key_async_from_dict():
    await test_create_hcx_activation_key_async(request_type=dict)

def test_create_hcx_activation_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateHcxActivationKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_hcx_activation_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_hcx_activation_key_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateHcxActivationKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_hcx_activation_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_hcx_activation_key_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_hcx_activation_key(parent='parent_value', hcx_activation_key=vmwareengine_resources.HcxActivationKey(name='name_value'), hcx_activation_key_id='hcx_activation_key_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].hcx_activation_key
        mock_val = vmwareengine_resources.HcxActivationKey(name='name_value')
        assert arg == mock_val
        arg = args[0].hcx_activation_key_id
        mock_val = 'hcx_activation_key_id_value'
        assert arg == mock_val

def test_create_hcx_activation_key_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_hcx_activation_key(vmwareengine.CreateHcxActivationKeyRequest(), parent='parent_value', hcx_activation_key=vmwareengine_resources.HcxActivationKey(name='name_value'), hcx_activation_key_id='hcx_activation_key_id_value')

@pytest.mark.asyncio
async def test_create_hcx_activation_key_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_hcx_activation_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_hcx_activation_key(parent='parent_value', hcx_activation_key=vmwareengine_resources.HcxActivationKey(name='name_value'), hcx_activation_key_id='hcx_activation_key_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].hcx_activation_key
        mock_val = vmwareengine_resources.HcxActivationKey(name='name_value')
        assert arg == mock_val
        arg = args[0].hcx_activation_key_id
        mock_val = 'hcx_activation_key_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_hcx_activation_key_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_hcx_activation_key(vmwareengine.CreateHcxActivationKeyRequest(), parent='parent_value', hcx_activation_key=vmwareengine_resources.HcxActivationKey(name='name_value'), hcx_activation_key_id='hcx_activation_key_id_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListHcxActivationKeysRequest, dict])
def test_list_hcx_activation_keys(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.return_value = vmwareengine.ListHcxActivationKeysResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_hcx_activation_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListHcxActivationKeysRequest()
    assert isinstance(response, pagers.ListHcxActivationKeysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_hcx_activation_keys_empty_call():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        client.list_hcx_activation_keys()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListHcxActivationKeysRequest()

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListHcxActivationKeysRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListHcxActivationKeysResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_hcx_activation_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListHcxActivationKeysRequest()
    assert isinstance(response, pagers.ListHcxActivationKeysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_async_from_dict():
    await test_list_hcx_activation_keys_async(request_type=dict)

def test_list_hcx_activation_keys_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListHcxActivationKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.return_value = vmwareengine.ListHcxActivationKeysResponse()
        client.list_hcx_activation_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListHcxActivationKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListHcxActivationKeysResponse())
        await client.list_hcx_activation_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_hcx_activation_keys_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.return_value = vmwareengine.ListHcxActivationKeysResponse()
        client.list_hcx_activation_keys(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_hcx_activation_keys_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_hcx_activation_keys(vmwareengine.ListHcxActivationKeysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.return_value = vmwareengine.ListHcxActivationKeysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListHcxActivationKeysResponse())
        response = await client.list_hcx_activation_keys(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_hcx_activation_keys(vmwareengine.ListHcxActivationKeysRequest(), parent='parent_value')

def test_list_hcx_activation_keys_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.side_effect = (vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()], next_page_token='abc'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[], next_page_token='def'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey()], next_page_token='ghi'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_hcx_activation_keys(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.HcxActivationKey) for i in results))

def test_list_hcx_activation_keys_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__') as call:
        call.side_effect = (vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()], next_page_token='abc'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[], next_page_token='def'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey()], next_page_token='ghi'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()]), RuntimeError)
        pages = list(client.list_hcx_activation_keys(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()], next_page_token='abc'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[], next_page_token='def'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey()], next_page_token='ghi'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()]), RuntimeError)
        async_pager = await client.list_hcx_activation_keys(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.HcxActivationKey) for i in responses))

@pytest.mark.asyncio
async def test_list_hcx_activation_keys_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_hcx_activation_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()], next_page_token='abc'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[], next_page_token='def'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey()], next_page_token='ghi'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_hcx_activation_keys(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetHcxActivationKeyRequest, dict])
def test_get_hcx_activation_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        call.return_value = vmwareengine_resources.HcxActivationKey(name='name_value', state=vmwareengine_resources.HcxActivationKey.State.AVAILABLE, activation_key='activation_key_value', uid='uid_value')
        response = client.get_hcx_activation_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetHcxActivationKeyRequest()
    assert isinstance(response, vmwareengine_resources.HcxActivationKey)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.HcxActivationKey.State.AVAILABLE
    assert response.activation_key == 'activation_key_value'
    assert response.uid == 'uid_value'

def test_get_hcx_activation_key_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        client.get_hcx_activation_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetHcxActivationKeyRequest()

@pytest.mark.asyncio
async def test_get_hcx_activation_key_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetHcxActivationKeyRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.HcxActivationKey(name='name_value', state=vmwareengine_resources.HcxActivationKey.State.AVAILABLE, activation_key='activation_key_value', uid='uid_value'))
        response = await client.get_hcx_activation_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetHcxActivationKeyRequest()
    assert isinstance(response, vmwareengine_resources.HcxActivationKey)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.HcxActivationKey.State.AVAILABLE
    assert response.activation_key == 'activation_key_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_hcx_activation_key_async_from_dict():
    await test_get_hcx_activation_key_async(request_type=dict)

def test_get_hcx_activation_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetHcxActivationKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        call.return_value = vmwareengine_resources.HcxActivationKey()
        client.get_hcx_activation_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_hcx_activation_key_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetHcxActivationKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.HcxActivationKey())
        await client.get_hcx_activation_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_hcx_activation_key_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        call.return_value = vmwareengine_resources.HcxActivationKey()
        client.get_hcx_activation_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_hcx_activation_key_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_hcx_activation_key(vmwareengine.GetHcxActivationKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_hcx_activation_key_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_hcx_activation_key), '__call__') as call:
        call.return_value = vmwareengine_resources.HcxActivationKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.HcxActivationKey())
        response = await client.get_hcx_activation_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_hcx_activation_key_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_hcx_activation_key(vmwareengine.GetHcxActivationKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.GetNetworkPolicyRequest, dict])
def test_get_network_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        call.return_value = vmwareengine_resources.NetworkPolicy(name='name_value', edge_services_cidr='edge_services_cidr_value', uid='uid_value', vmware_engine_network='vmware_engine_network_value', description='description_value', vmware_engine_network_canonical='vmware_engine_network_canonical_value')
        response = client.get_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetNetworkPolicyRequest()
    assert isinstance(response, vmwareengine_resources.NetworkPolicy)
    assert response.name == 'name_value'
    assert response.edge_services_cidr == 'edge_services_cidr_value'
    assert response.uid == 'uid_value'
    assert response.vmware_engine_network == 'vmware_engine_network_value'
    assert response.description == 'description_value'
    assert response.vmware_engine_network_canonical == 'vmware_engine_network_canonical_value'

def test_get_network_policy_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        client.get_network_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetNetworkPolicyRequest()

@pytest.mark.asyncio
async def test_get_network_policy_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetNetworkPolicyRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.NetworkPolicy(name='name_value', edge_services_cidr='edge_services_cidr_value', uid='uid_value', vmware_engine_network='vmware_engine_network_value', description='description_value', vmware_engine_network_canonical='vmware_engine_network_canonical_value'))
        response = await client.get_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetNetworkPolicyRequest()
    assert isinstance(response, vmwareengine_resources.NetworkPolicy)
    assert response.name == 'name_value'
    assert response.edge_services_cidr == 'edge_services_cidr_value'
    assert response.uid == 'uid_value'
    assert response.vmware_engine_network == 'vmware_engine_network_value'
    assert response.description == 'description_value'
    assert response.vmware_engine_network_canonical == 'vmware_engine_network_canonical_value'

@pytest.mark.asyncio
async def test_get_network_policy_async_from_dict():
    await test_get_network_policy_async(request_type=dict)

def test_get_network_policy_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetNetworkPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        call.return_value = vmwareengine_resources.NetworkPolicy()
        client.get_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_network_policy_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetNetworkPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.NetworkPolicy())
        await client.get_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_network_policy_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        call.return_value = vmwareengine_resources.NetworkPolicy()
        client.get_network_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_network_policy_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_network_policy(vmwareengine.GetNetworkPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_network_policy_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_network_policy), '__call__') as call:
        call.return_value = vmwareengine_resources.NetworkPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.NetworkPolicy())
        response = await client.get_network_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_network_policy_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_network_policy(vmwareengine.GetNetworkPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListNetworkPoliciesRequest, dict])
def test_list_network_policies(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.return_value = vmwareengine.ListNetworkPoliciesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_network_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListNetworkPoliciesRequest()
    assert isinstance(response, pagers.ListNetworkPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_network_policies_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        client.list_network_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListNetworkPoliciesRequest()

@pytest.mark.asyncio
async def test_list_network_policies_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListNetworkPoliciesRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListNetworkPoliciesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_network_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListNetworkPoliciesRequest()
    assert isinstance(response, pagers.ListNetworkPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_network_policies_async_from_dict():
    await test_list_network_policies_async(request_type=dict)

def test_list_network_policies_field_headers():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListNetworkPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.return_value = vmwareengine.ListNetworkPoliciesResponse()
        client.list_network_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_network_policies_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListNetworkPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListNetworkPoliciesResponse())
        await client.list_network_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_network_policies_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.return_value = vmwareengine.ListNetworkPoliciesResponse()
        client.list_network_policies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_network_policies_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_network_policies(vmwareengine.ListNetworkPoliciesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_network_policies_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.return_value = vmwareengine.ListNetworkPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListNetworkPoliciesResponse())
        response = await client.list_network_policies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_network_policies_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_network_policies(vmwareengine.ListNetworkPoliciesRequest(), parent='parent_value')

def test_list_network_policies_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.side_effect = (vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()], next_page_token='abc'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[], next_page_token='def'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy()], next_page_token='ghi'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_network_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.NetworkPolicy) for i in results))

def test_list_network_policies_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_network_policies), '__call__') as call:
        call.side_effect = (vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()], next_page_token='abc'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[], next_page_token='def'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy()], next_page_token='ghi'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()]), RuntimeError)
        pages = list(client.list_network_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_network_policies_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_network_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()], next_page_token='abc'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[], next_page_token='def'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy()], next_page_token='ghi'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()]), RuntimeError)
        async_pager = await client.list_network_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.NetworkPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_network_policies_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_network_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()], next_page_token='abc'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[], next_page_token='def'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy()], next_page_token='ghi'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_network_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.CreateNetworkPolicyRequest, dict])
def test_create_network_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateNetworkPolicyRequest()
    assert isinstance(response, future.Future)

def test_create_network_policy_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        client.create_network_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateNetworkPolicyRequest()

@pytest.mark.asyncio
async def test_create_network_policy_async(transport: str='grpc_asyncio', request_type=vmwareengine.CreateNetworkPolicyRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateNetworkPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_network_policy_async_from_dict():
    await test_create_network_policy_async(request_type=dict)

def test_create_network_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateNetworkPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_network_policy_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateNetworkPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_network_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_network_policy(parent='parent_value', network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), network_policy_id='network_policy_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].network_policy
        mock_val = vmwareengine_resources.NetworkPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].network_policy_id
        mock_val = 'network_policy_id_value'
        assert arg == mock_val

def test_create_network_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_network_policy(vmwareengine.CreateNetworkPolicyRequest(), parent='parent_value', network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), network_policy_id='network_policy_id_value')

@pytest.mark.asyncio
async def test_create_network_policy_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_network_policy(parent='parent_value', network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), network_policy_id='network_policy_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].network_policy
        mock_val = vmwareengine_resources.NetworkPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].network_policy_id
        mock_val = 'network_policy_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_network_policy_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_network_policy(vmwareengine.CreateNetworkPolicyRequest(), parent='parent_value', network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), network_policy_id='network_policy_id_value')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateNetworkPolicyRequest, dict])
def test_update_network_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateNetworkPolicyRequest()
    assert isinstance(response, future.Future)

def test_update_network_policy_empty_call():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        client.update_network_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateNetworkPolicyRequest()

@pytest.mark.asyncio
async def test_update_network_policy_async(transport: str='grpc_asyncio', request_type=vmwareengine.UpdateNetworkPolicyRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateNetworkPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_network_policy_async_from_dict():
    await test_update_network_policy_async(request_type=dict)

def test_update_network_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateNetworkPolicyRequest()
    request.network_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'network_policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_network_policy_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateNetworkPolicyRequest()
    request.network_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'network_policy.name=name_value') in kw['metadata']

def test_update_network_policy_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_network_policy(network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].network_policy
        mock_val = vmwareengine_resources.NetworkPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_network_policy_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_network_policy(vmwareengine.UpdateNetworkPolicyRequest(), network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_network_policy_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_network_policy(network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].network_policy
        mock_val = vmwareengine_resources.NetworkPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_network_policy_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_network_policy(vmwareengine.UpdateNetworkPolicyRequest(), network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmwareengine.DeleteNetworkPolicyRequest, dict])
def test_delete_network_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteNetworkPolicyRequest()
    assert isinstance(response, future.Future)

def test_delete_network_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        client.delete_network_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteNetworkPolicyRequest()

@pytest.mark.asyncio
async def test_delete_network_policy_async(transport: str='grpc_asyncio', request_type=vmwareengine.DeleteNetworkPolicyRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteNetworkPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_network_policy_async_from_dict():
    await test_delete_network_policy_async(request_type=dict)

def test_delete_network_policy_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeleteNetworkPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_network_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_network_policy_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeleteNetworkPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_network_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_network_policy_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_network_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_network_policy_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_network_policy(vmwareengine.DeleteNetworkPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_network_policy_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_network_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_network_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_network_policy_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_network_policy(vmwareengine.DeleteNetworkPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.CreateVmwareEngineNetworkRequest, dict])
def test_create_vmware_engine_network(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateVmwareEngineNetworkRequest()
    assert isinstance(response, future.Future)

def test_create_vmware_engine_network_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        client.create_vmware_engine_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateVmwareEngineNetworkRequest()

@pytest.mark.asyncio
async def test_create_vmware_engine_network_async(transport: str='grpc_asyncio', request_type=vmwareengine.CreateVmwareEngineNetworkRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreateVmwareEngineNetworkRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_vmware_engine_network_async_from_dict():
    await test_create_vmware_engine_network_async(request_type=dict)

def test_create_vmware_engine_network_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateVmwareEngineNetworkRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_vmware_engine_network_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreateVmwareEngineNetworkRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_vmware_engine_network_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_vmware_engine_network(parent='parent_value', vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), vmware_engine_network_id='vmware_engine_network_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].vmware_engine_network
        mock_val = vmwareengine_resources.VmwareEngineNetwork(name='name_value')
        assert arg == mock_val
        arg = args[0].vmware_engine_network_id
        mock_val = 'vmware_engine_network_id_value'
        assert arg == mock_val

def test_create_vmware_engine_network_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_vmware_engine_network(vmwareengine.CreateVmwareEngineNetworkRequest(), parent='parent_value', vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), vmware_engine_network_id='vmware_engine_network_id_value')

@pytest.mark.asyncio
async def test_create_vmware_engine_network_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_vmware_engine_network(parent='parent_value', vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), vmware_engine_network_id='vmware_engine_network_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].vmware_engine_network
        mock_val = vmwareengine_resources.VmwareEngineNetwork(name='name_value')
        assert arg == mock_val
        arg = args[0].vmware_engine_network_id
        mock_val = 'vmware_engine_network_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_vmware_engine_network_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_vmware_engine_network(vmwareengine.CreateVmwareEngineNetworkRequest(), parent='parent_value', vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), vmware_engine_network_id='vmware_engine_network_id_value')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateVmwareEngineNetworkRequest, dict])
def test_update_vmware_engine_network(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateVmwareEngineNetworkRequest()
    assert isinstance(response, future.Future)

def test_update_vmware_engine_network_empty_call():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        client.update_vmware_engine_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateVmwareEngineNetworkRequest()

@pytest.mark.asyncio
async def test_update_vmware_engine_network_async(transport: str='grpc_asyncio', request_type=vmwareengine.UpdateVmwareEngineNetworkRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdateVmwareEngineNetworkRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_vmware_engine_network_async_from_dict():
    await test_update_vmware_engine_network_async(request_type=dict)

def test_update_vmware_engine_network_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateVmwareEngineNetworkRequest()
    request.vmware_engine_network.name = 'name_value'
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'vmware_engine_network.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_vmware_engine_network_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdateVmwareEngineNetworkRequest()
    request.vmware_engine_network.name = 'name_value'
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'vmware_engine_network.name=name_value') in kw['metadata']

def test_update_vmware_engine_network_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_vmware_engine_network(vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].vmware_engine_network
        mock_val = vmwareengine_resources.VmwareEngineNetwork(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_vmware_engine_network_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_vmware_engine_network(vmwareengine.UpdateVmwareEngineNetworkRequest(), vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_vmware_engine_network_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_vmware_engine_network(vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].vmware_engine_network
        mock_val = vmwareengine_resources.VmwareEngineNetwork(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_vmware_engine_network_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_vmware_engine_network(vmwareengine.UpdateVmwareEngineNetworkRequest(), vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmwareengine.DeleteVmwareEngineNetworkRequest, dict])
def test_delete_vmware_engine_network(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteVmwareEngineNetworkRequest()
    assert isinstance(response, future.Future)

def test_delete_vmware_engine_network_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        client.delete_vmware_engine_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteVmwareEngineNetworkRequest()

@pytest.mark.asyncio
async def test_delete_vmware_engine_network_async(transport: str='grpc_asyncio', request_type=vmwareengine.DeleteVmwareEngineNetworkRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeleteVmwareEngineNetworkRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_vmware_engine_network_async_from_dict():
    await test_delete_vmware_engine_network_async(request_type=dict)

def test_delete_vmware_engine_network_field_headers():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeleteVmwareEngineNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_vmware_engine_network_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeleteVmwareEngineNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_vmware_engine_network_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_vmware_engine_network(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_vmware_engine_network_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_vmware_engine_network(vmwareengine.DeleteVmwareEngineNetworkRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_vmware_engine_network_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_vmware_engine_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_vmware_engine_network(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_vmware_engine_network_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_vmware_engine_network(vmwareengine.DeleteVmwareEngineNetworkRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.GetVmwareEngineNetworkRequest, dict])
def test_get_vmware_engine_network(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        call.return_value = vmwareengine_resources.VmwareEngineNetwork(name='name_value', description='description_value', state=vmwareengine_resources.VmwareEngineNetwork.State.CREATING, type_=vmwareengine_resources.VmwareEngineNetwork.Type.LEGACY, uid='uid_value', etag='etag_value')
        response = client.get_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetVmwareEngineNetworkRequest()
    assert isinstance(response, vmwareengine_resources.VmwareEngineNetwork)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == vmwareengine_resources.VmwareEngineNetwork.State.CREATING
    assert response.type_ == vmwareengine_resources.VmwareEngineNetwork.Type.LEGACY
    assert response.uid == 'uid_value'
    assert response.etag == 'etag_value'

def test_get_vmware_engine_network_empty_call():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        client.get_vmware_engine_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetVmwareEngineNetworkRequest()

@pytest.mark.asyncio
async def test_get_vmware_engine_network_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetVmwareEngineNetworkRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.VmwareEngineNetwork(name='name_value', description='description_value', state=vmwareengine_resources.VmwareEngineNetwork.State.CREATING, type_=vmwareengine_resources.VmwareEngineNetwork.Type.LEGACY, uid='uid_value', etag='etag_value'))
        response = await client.get_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetVmwareEngineNetworkRequest()
    assert isinstance(response, vmwareengine_resources.VmwareEngineNetwork)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == vmwareengine_resources.VmwareEngineNetwork.State.CREATING
    assert response.type_ == vmwareengine_resources.VmwareEngineNetwork.Type.LEGACY
    assert response.uid == 'uid_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_vmware_engine_network_async_from_dict():
    await test_get_vmware_engine_network_async(request_type=dict)

def test_get_vmware_engine_network_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetVmwareEngineNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        call.return_value = vmwareengine_resources.VmwareEngineNetwork()
        client.get_vmware_engine_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_vmware_engine_network_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetVmwareEngineNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.VmwareEngineNetwork())
        await client.get_vmware_engine_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_vmware_engine_network_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        call.return_value = vmwareengine_resources.VmwareEngineNetwork()
        client.get_vmware_engine_network(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_vmware_engine_network_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_vmware_engine_network(vmwareengine.GetVmwareEngineNetworkRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_vmware_engine_network_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vmware_engine_network), '__call__') as call:
        call.return_value = vmwareengine_resources.VmwareEngineNetwork()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.VmwareEngineNetwork())
        response = await client.get_vmware_engine_network(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_vmware_engine_network_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_vmware_engine_network(vmwareengine.GetVmwareEngineNetworkRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListVmwareEngineNetworksRequest, dict])
def test_list_vmware_engine_networks(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.return_value = vmwareengine.ListVmwareEngineNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_vmware_engine_networks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListVmwareEngineNetworksRequest()
    assert isinstance(response, pagers.ListVmwareEngineNetworksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_vmware_engine_networks_empty_call():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        client.list_vmware_engine_networks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListVmwareEngineNetworksRequest()

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListVmwareEngineNetworksRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListVmwareEngineNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_vmware_engine_networks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListVmwareEngineNetworksRequest()
    assert isinstance(response, pagers.ListVmwareEngineNetworksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_async_from_dict():
    await test_list_vmware_engine_networks_async(request_type=dict)

def test_list_vmware_engine_networks_field_headers():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListVmwareEngineNetworksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.return_value = vmwareengine.ListVmwareEngineNetworksResponse()
        client.list_vmware_engine_networks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListVmwareEngineNetworksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListVmwareEngineNetworksResponse())
        await client.list_vmware_engine_networks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_vmware_engine_networks_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.return_value = vmwareengine.ListVmwareEngineNetworksResponse()
        client.list_vmware_engine_networks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_vmware_engine_networks_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_vmware_engine_networks(vmwareengine.ListVmwareEngineNetworksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.return_value = vmwareengine.ListVmwareEngineNetworksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListVmwareEngineNetworksResponse())
        response = await client.list_vmware_engine_networks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_vmware_engine_networks(vmwareengine.ListVmwareEngineNetworksRequest(), parent='parent_value')

def test_list_vmware_engine_networks_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.side_effect = (vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()], next_page_token='abc'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[], next_page_token='def'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork()], next_page_token='ghi'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_vmware_engine_networks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.VmwareEngineNetwork) for i in results))

def test_list_vmware_engine_networks_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__') as call:
        call.side_effect = (vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()], next_page_token='abc'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[], next_page_token='def'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork()], next_page_token='ghi'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()]), RuntimeError)
        pages = list(client.list_vmware_engine_networks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()], next_page_token='abc'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[], next_page_token='def'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork()], next_page_token='ghi'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()]), RuntimeError)
        async_pager = await client.list_vmware_engine_networks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.VmwareEngineNetwork) for i in responses))

@pytest.mark.asyncio
async def test_list_vmware_engine_networks_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vmware_engine_networks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()], next_page_token='abc'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[], next_page_token='def'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork()], next_page_token='ghi'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_vmware_engine_networks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.CreatePrivateConnectionRequest, dict])
def test_create_private_connection(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreatePrivateConnectionRequest()
    assert isinstance(response, future.Future)

def test_create_private_connection_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        client.create_private_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreatePrivateConnectionRequest()

@pytest.mark.asyncio
async def test_create_private_connection_async(transport: str='grpc_asyncio', request_type=vmwareengine.CreatePrivateConnectionRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.CreatePrivateConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_private_connection_async_from_dict():
    await test_create_private_connection_async(request_type=dict)

def test_create_private_connection_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreatePrivateConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_private_connection_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.CreatePrivateConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_private_connection_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_private_connection(parent='parent_value', private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), private_connection_id='private_connection_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].private_connection
        mock_val = vmwareengine_resources.PrivateConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].private_connection_id
        mock_val = 'private_connection_id_value'
        assert arg == mock_val

def test_create_private_connection_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_private_connection(vmwareengine.CreatePrivateConnectionRequest(), parent='parent_value', private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), private_connection_id='private_connection_id_value')

@pytest.mark.asyncio
async def test_create_private_connection_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_private_connection(parent='parent_value', private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), private_connection_id='private_connection_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].private_connection
        mock_val = vmwareengine_resources.PrivateConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].private_connection_id
        mock_val = 'private_connection_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_private_connection_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_private_connection(vmwareengine.CreatePrivateConnectionRequest(), parent='parent_value', private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), private_connection_id='private_connection_id_value')

@pytest.mark.parametrize('request_type', [vmwareengine.GetPrivateConnectionRequest, dict])
def test_get_private_connection(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateConnection(name='name_value', description='description_value', state=vmwareengine_resources.PrivateConnection.State.CREATING, vmware_engine_network='vmware_engine_network_value', vmware_engine_network_canonical='vmware_engine_network_canonical_value', type_=vmwareengine_resources.PrivateConnection.Type.PRIVATE_SERVICE_ACCESS, peering_id='peering_id_value', routing_mode=vmwareengine_resources.PrivateConnection.RoutingMode.GLOBAL, uid='uid_value', service_network='service_network_value', peering_state=vmwareengine_resources.PrivateConnection.PeeringState.PEERING_ACTIVE)
        response = client.get_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetPrivateConnectionRequest()
    assert isinstance(response, vmwareengine_resources.PrivateConnection)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == vmwareengine_resources.PrivateConnection.State.CREATING
    assert response.vmware_engine_network == 'vmware_engine_network_value'
    assert response.vmware_engine_network_canonical == 'vmware_engine_network_canonical_value'
    assert response.type_ == vmwareengine_resources.PrivateConnection.Type.PRIVATE_SERVICE_ACCESS
    assert response.peering_id == 'peering_id_value'
    assert response.routing_mode == vmwareengine_resources.PrivateConnection.RoutingMode.GLOBAL
    assert response.uid == 'uid_value'
    assert response.service_network == 'service_network_value'
    assert response.peering_state == vmwareengine_resources.PrivateConnection.PeeringState.PEERING_ACTIVE

def test_get_private_connection_empty_call():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        client.get_private_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetPrivateConnectionRequest()

@pytest.mark.asyncio
async def test_get_private_connection_async(transport: str='grpc_asyncio', request_type=vmwareengine.GetPrivateConnectionRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.PrivateConnection(name='name_value', description='description_value', state=vmwareengine_resources.PrivateConnection.State.CREATING, vmware_engine_network='vmware_engine_network_value', vmware_engine_network_canonical='vmware_engine_network_canonical_value', type_=vmwareengine_resources.PrivateConnection.Type.PRIVATE_SERVICE_ACCESS, peering_id='peering_id_value', routing_mode=vmwareengine_resources.PrivateConnection.RoutingMode.GLOBAL, uid='uid_value', service_network='service_network_value', peering_state=vmwareengine_resources.PrivateConnection.PeeringState.PEERING_ACTIVE))
        response = await client.get_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.GetPrivateConnectionRequest()
    assert isinstance(response, vmwareengine_resources.PrivateConnection)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == vmwareengine_resources.PrivateConnection.State.CREATING
    assert response.vmware_engine_network == 'vmware_engine_network_value'
    assert response.vmware_engine_network_canonical == 'vmware_engine_network_canonical_value'
    assert response.type_ == vmwareengine_resources.PrivateConnection.Type.PRIVATE_SERVICE_ACCESS
    assert response.peering_id == 'peering_id_value'
    assert response.routing_mode == vmwareengine_resources.PrivateConnection.RoutingMode.GLOBAL
    assert response.uid == 'uid_value'
    assert response.service_network == 'service_network_value'
    assert response.peering_state == vmwareengine_resources.PrivateConnection.PeeringState.PEERING_ACTIVE

@pytest.mark.asyncio
async def test_get_private_connection_async_from_dict():
    await test_get_private_connection_async(request_type=dict)

def test_get_private_connection_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetPrivateConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateConnection()
        client.get_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_private_connection_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.GetPrivateConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.PrivateConnection())
        await client.get_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_private_connection_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateConnection()
        client.get_private_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_private_connection_flattened_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_private_connection(vmwareengine.GetPrivateConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_private_connection_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_private_connection), '__call__') as call:
        call.return_value = vmwareengine_resources.PrivateConnection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine_resources.PrivateConnection())
        response = await client.get_private_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_private_connection_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_private_connection(vmwareengine.GetPrivateConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListPrivateConnectionsRequest, dict])
def test_list_private_connections(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_private_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateConnectionsRequest()
    assert isinstance(response, pagers.ListPrivateConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_private_connections_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        client.list_private_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateConnectionsRequest()

@pytest.mark.asyncio
async def test_list_private_connections_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListPrivateConnectionsRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_private_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateConnectionsRequest()
    assert isinstance(response, pagers.ListPrivateConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_private_connections_async_from_dict():
    await test_list_private_connections_async(request_type=dict)

def test_list_private_connections_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListPrivateConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionsResponse()
        client.list_private_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_private_connections_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListPrivateConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateConnectionsResponse())
        await client.list_private_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_private_connections_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionsResponse()
        client.list_private_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_private_connections_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_private_connections(vmwareengine.ListPrivateConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_private_connections_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateConnectionsResponse())
        response = await client.list_private_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_private_connections_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_private_connections(vmwareengine.ListPrivateConnectionsRequest(), parent='parent_value')

def test_list_private_connections_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()], next_page_token='abc'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[], next_page_token='def'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_private_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.PrivateConnection) for i in results))

def test_list_private_connections_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_private_connections), '__call__') as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()], next_page_token='abc'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[], next_page_token='def'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()]), RuntimeError)
        pages = list(client.list_private_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_private_connections_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_private_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()], next_page_token='abc'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[], next_page_token='def'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()]), RuntimeError)
        async_pager = await client.list_private_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.PrivateConnection) for i in responses))

@pytest.mark.asyncio
async def test_list_private_connections_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_private_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()], next_page_token='abc'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[], next_page_token='def'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_private_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.UpdatePrivateConnectionRequest, dict])
def test_update_private_connection(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdatePrivateConnectionRequest()
    assert isinstance(response, future.Future)

def test_update_private_connection_empty_call():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        client.update_private_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdatePrivateConnectionRequest()

@pytest.mark.asyncio
async def test_update_private_connection_async(transport: str='grpc_asyncio', request_type=vmwareengine.UpdatePrivateConnectionRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.UpdatePrivateConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_private_connection_async_from_dict():
    await test_update_private_connection_async(request_type=dict)

def test_update_private_connection_field_headers():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdatePrivateConnectionRequest()
    request.private_connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_connection.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_private_connection_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.UpdatePrivateConnectionRequest()
    request.private_connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'private_connection.name=name_value') in kw['metadata']

def test_update_private_connection_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_private_connection(private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_connection
        mock_val = vmwareengine_resources.PrivateConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_private_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_private_connection(vmwareengine.UpdatePrivateConnectionRequest(), private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_private_connection_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_private_connection(private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].private_connection
        mock_val = vmwareengine_resources.PrivateConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_private_connection_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_private_connection(vmwareengine.UpdatePrivateConnectionRequest(), private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [vmwareengine.DeletePrivateConnectionRequest, dict])
def test_delete_private_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeletePrivateConnectionRequest()
    assert isinstance(response, future.Future)

def test_delete_private_connection_empty_call():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        client.delete_private_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeletePrivateConnectionRequest()

@pytest.mark.asyncio
async def test_delete_private_connection_async(transport: str='grpc_asyncio', request_type=vmwareengine.DeletePrivateConnectionRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.DeletePrivateConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_private_connection_async_from_dict():
    await test_delete_private_connection_async(request_type=dict)

def test_delete_private_connection_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeletePrivateConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_private_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_private_connection_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.DeletePrivateConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_private_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_private_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_private_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_private_connection_flattened_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_private_connection(vmwareengine.DeletePrivateConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_private_connection_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_private_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_private_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_private_connection_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_private_connection(vmwareengine.DeletePrivateConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vmwareengine.ListPrivateConnectionPeeringRoutesRequest, dict])
def test_list_private_connection_peering_routes(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse(next_page_token='next_page_token_value')
        response = client.list_private_connection_peering_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateConnectionPeeringRoutesRequest()
    assert isinstance(response, pagers.ListPrivateConnectionPeeringRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_private_connection_peering_routes_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        client.list_private_connection_peering_routes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateConnectionPeeringRoutesRequest()

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_async(transport: str='grpc_asyncio', request_type=vmwareengine.ListPrivateConnectionPeeringRoutesRequest):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateConnectionPeeringRoutesResponse(next_page_token='next_page_token_value'))
        response = await client.list_private_connection_peering_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vmwareengine.ListPrivateConnectionPeeringRoutesRequest()
    assert isinstance(response, pagers.ListPrivateConnectionPeeringRoutesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_async_from_dict():
    await test_list_private_connection_peering_routes_async(request_type=dict)

def test_list_private_connection_peering_routes_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListPrivateConnectionPeeringRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse()
        client.list_private_connection_peering_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vmwareengine.ListPrivateConnectionPeeringRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateConnectionPeeringRoutesResponse())
        await client.list_private_connection_peering_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_private_connection_peering_routes_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse()
        client.list_private_connection_peering_routes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_private_connection_peering_routes_flattened_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_private_connection_peering_routes(vmwareengine.ListPrivateConnectionPeeringRoutesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_flattened_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vmwareengine.ListPrivateConnectionPeeringRoutesResponse())
        response = await client.list_private_connection_peering_routes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_flattened_error_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_private_connection_peering_routes(vmwareengine.ListPrivateConnectionPeeringRoutesRequest(), parent='parent_value')

def test_list_private_connection_peering_routes_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()], next_page_token='abc'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[], next_page_token='def'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_private_connection_peering_routes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.PeeringRoute) for i in results))

def test_list_private_connection_peering_routes_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__') as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()], next_page_token='abc'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[], next_page_token='def'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()]), RuntimeError)
        pages = list(client.list_private_connection_peering_routes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_async_pager():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()], next_page_token='abc'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[], next_page_token='def'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()]), RuntimeError)
        async_pager = await client.list_private_connection_peering_routes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vmwareengine_resources.PeeringRoute) for i in responses))

@pytest.mark.asyncio
async def test_list_private_connection_peering_routes_async_pages():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_private_connection_peering_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()], next_page_token='abc'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[], next_page_token='def'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_private_connection_peering_routes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.ListPrivateCloudsRequest, dict])
def test_list_private_clouds_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListPrivateCloudsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListPrivateCloudsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_private_clouds(request)
    assert isinstance(response, pagers.ListPrivateCloudsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_private_clouds_rest_required_fields(request_type=vmwareengine.ListPrivateCloudsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_private_clouds._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_private_clouds._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListPrivateCloudsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListPrivateCloudsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_private_clouds(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_private_clouds_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_private_clouds._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_private_clouds_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_private_clouds') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_private_clouds') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListPrivateCloudsRequest.pb(vmwareengine.ListPrivateCloudsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListPrivateCloudsResponse.to_json(vmwareengine.ListPrivateCloudsResponse())
        request = vmwareengine.ListPrivateCloudsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListPrivateCloudsResponse()
        client.list_private_clouds(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_private_clouds_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListPrivateCloudsRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_private_clouds(request)

def test_list_private_clouds_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListPrivateCloudsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListPrivateCloudsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_private_clouds(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/privateClouds' % client.transport._host, args[1])

def test_list_private_clouds_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_private_clouds(vmwareengine.ListPrivateCloudsRequest(), parent='parent_value')

def test_list_private_clouds_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()], next_page_token='abc'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[], next_page_token='def'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud()], next_page_token='ghi'), vmwareengine.ListPrivateCloudsResponse(private_clouds=[vmwareengine_resources.PrivateCloud(), vmwareengine_resources.PrivateCloud()]))
        response = response + response
        response = tuple((vmwareengine.ListPrivateCloudsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_private_clouds(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.PrivateCloud) for i in results))
        pages = list(client.list_private_clouds(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetPrivateCloudRequest, dict])
def test_get_private_cloud_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.PrivateCloud(name='name_value', state=vmwareengine_resources.PrivateCloud.State.ACTIVE, description='description_value', uid='uid_value', type_=vmwareengine_resources.PrivateCloud.Type.TIME_LIMITED)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.PrivateCloud.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_private_cloud(request)
    assert isinstance(response, vmwareengine_resources.PrivateCloud)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.PrivateCloud.State.ACTIVE
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.type_ == vmwareengine_resources.PrivateCloud.Type.TIME_LIMITED

def test_get_private_cloud_rest_required_fields(request_type=vmwareengine.GetPrivateCloudRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.PrivateCloud()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.PrivateCloud.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_private_cloud(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_private_cloud_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_private_cloud._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_private_cloud_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_private_cloud') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_private_cloud') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetPrivateCloudRequest.pb(vmwareengine.GetPrivateCloudRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.PrivateCloud.to_json(vmwareengine_resources.PrivateCloud())
        request = vmwareengine.GetPrivateCloudRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.PrivateCloud()
        client.get_private_cloud(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_private_cloud_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetPrivateCloudRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_private_cloud(request)

def test_get_private_cloud_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.PrivateCloud()
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.PrivateCloud.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_private_cloud(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*}' % client.transport._host, args[1])

def test_get_private_cloud_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_private_cloud(vmwareengine.GetPrivateCloudRequest(), name='name_value')

def test_get_private_cloud_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.CreatePrivateCloudRequest, dict])
def test_create_private_cloud_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['private_cloud'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}, 'state': 1, 'network_config': {'management_cidr': 'management_cidr_value', 'vmware_engine_network': 'vmware_engine_network_value', 'vmware_engine_network_canonical': 'vmware_engine_network_canonical_value', 'management_ip_address_layout_version': 3836}, 'management_cluster': {'cluster_id': 'cluster_id_value', 'node_type_configs': {}}, 'description': 'description_value', 'hcx': {'internal_ip': 'internal_ip_value', 'version': 'version_value', 'state': 1, 'fqdn': 'fqdn_value'}, 'nsx': {'internal_ip': 'internal_ip_value', 'version': 'version_value', 'state': 1, 'fqdn': 'fqdn_value'}, 'vcenter': {'internal_ip': 'internal_ip_value', 'version': 'version_value', 'state': 1, 'fqdn': 'fqdn_value'}, 'uid': 'uid_value', 'type_': 1}
    test_field = vmwareengine.CreatePrivateCloudRequest.meta.fields['private_cloud']

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
    for (field, value) in request_init['private_cloud'].items():
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
                for i in range(0, len(request_init['private_cloud'][field])):
                    del request_init['private_cloud'][field][i][subfield]
            else:
                del request_init['private_cloud'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_private_cloud(request)
    assert response.operation.name == 'operations/spam'

def test_create_private_cloud_rest_required_fields(request_type=vmwareengine.CreatePrivateCloudRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['private_cloud_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'privateCloudId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'privateCloudId' in jsonified_request
    assert jsonified_request['privateCloudId'] == request_init['private_cloud_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['privateCloudId'] = 'private_cloud_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_private_cloud._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('private_cloud_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'privateCloudId' in jsonified_request
    assert jsonified_request['privateCloudId'] == 'private_cloud_id_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_private_cloud(request)
            expected_params = [('privateCloudId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_private_cloud_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_private_cloud._get_unset_required_fields({})
    assert set(unset_fields) == set(('privateCloudId', 'requestId', 'validateOnly')) & set(('parent', 'privateCloudId', 'privateCloud'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_private_cloud_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_create_private_cloud') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_create_private_cloud') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.CreatePrivateCloudRequest.pb(vmwareengine.CreatePrivateCloudRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.CreatePrivateCloudRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_private_cloud(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_private_cloud_rest_bad_request(transport: str='rest', request_type=vmwareengine.CreatePrivateCloudRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_private_cloud(request)

def test_create_private_cloud_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), private_cloud_id='private_cloud_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_private_cloud(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/privateClouds' % client.transport._host, args[1])

def test_create_private_cloud_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_private_cloud(vmwareengine.CreatePrivateCloudRequest(), parent='parent_value', private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), private_cloud_id='private_cloud_id_value')

def test_create_private_cloud_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdatePrivateCloudRequest, dict])
def test_update_private_cloud_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'private_cloud': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}}
    request_init['private_cloud'] = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'expire_time': {}, 'state': 1, 'network_config': {'management_cidr': 'management_cidr_value', 'vmware_engine_network': 'vmware_engine_network_value', 'vmware_engine_network_canonical': 'vmware_engine_network_canonical_value', 'management_ip_address_layout_version': 3836}, 'management_cluster': {'cluster_id': 'cluster_id_value', 'node_type_configs': {}}, 'description': 'description_value', 'hcx': {'internal_ip': 'internal_ip_value', 'version': 'version_value', 'state': 1, 'fqdn': 'fqdn_value'}, 'nsx': {'internal_ip': 'internal_ip_value', 'version': 'version_value', 'state': 1, 'fqdn': 'fqdn_value'}, 'vcenter': {'internal_ip': 'internal_ip_value', 'version': 'version_value', 'state': 1, 'fqdn': 'fqdn_value'}, 'uid': 'uid_value', 'type_': 1}
    test_field = vmwareengine.UpdatePrivateCloudRequest.meta.fields['private_cloud']

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
    for (field, value) in request_init['private_cloud'].items():
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
                for i in range(0, len(request_init['private_cloud'][field])):
                    del request_init['private_cloud'][field][i][subfield]
            else:
                del request_init['private_cloud'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_private_cloud(request)
    assert response.operation.name == 'operations/spam'

def test_update_private_cloud_rest_required_fields(request_type=vmwareengine.UpdatePrivateCloudRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_private_cloud._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_private_cloud(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_private_cloud_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_private_cloud._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('privateCloud', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_private_cloud_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_update_private_cloud') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_update_private_cloud') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UpdatePrivateCloudRequest.pb(vmwareengine.UpdatePrivateCloudRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UpdatePrivateCloudRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_private_cloud(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_private_cloud_rest_bad_request(transport: str='rest', request_type=vmwareengine.UpdatePrivateCloudRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'private_cloud': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_private_cloud(request)

def test_update_private_cloud_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'private_cloud': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}}
        mock_args = dict(private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_private_cloud(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{private_cloud.name=projects/*/locations/*/privateClouds/*}' % client.transport._host, args[1])

def test_update_private_cloud_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_private_cloud(vmwareengine.UpdatePrivateCloudRequest(), private_cloud=vmwareengine_resources.PrivateCloud(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_private_cloud_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.DeletePrivateCloudRequest, dict])
def test_delete_private_cloud_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_private_cloud(request)
    assert response.operation.name == 'operations/spam'

def test_delete_private_cloud_rest_required_fields(request_type=vmwareengine.DeletePrivateCloudRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_private_cloud._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('delay_hours', 'force', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_private_cloud(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_private_cloud_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_private_cloud._get_unset_required_fields({})
    assert set(unset_fields) == set(('delayHours', 'force', 'requestId')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_private_cloud_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_delete_private_cloud') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_delete_private_cloud') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.DeletePrivateCloudRequest.pb(vmwareengine.DeletePrivateCloudRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.DeletePrivateCloudRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_private_cloud(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_private_cloud_rest_bad_request(transport: str='rest', request_type=vmwareengine.DeletePrivateCloudRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_private_cloud(request)

def test_delete_private_cloud_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_private_cloud(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*}' % client.transport._host, args[1])

def test_delete_private_cloud_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_private_cloud(vmwareengine.DeletePrivateCloudRequest(), name='name_value')

def test_delete_private_cloud_rest_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.UndeletePrivateCloudRequest, dict])
def test_undelete_private_cloud_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.undelete_private_cloud(request)
    assert response.operation.name == 'operations/spam'

def test_undelete_private_cloud_rest_required_fields(request_type=vmwareengine.UndeletePrivateCloudRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undelete_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undelete_private_cloud._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.undelete_private_cloud(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_undelete_private_cloud_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.undelete_private_cloud._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_undelete_private_cloud_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_undelete_private_cloud') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_undelete_private_cloud') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UndeletePrivateCloudRequest.pb(vmwareengine.UndeletePrivateCloudRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UndeletePrivateCloudRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.undelete_private_cloud(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_undelete_private_cloud_rest_bad_request(transport: str='rest', request_type=vmwareengine.UndeletePrivateCloudRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.undelete_private_cloud(request)

def test_undelete_private_cloud_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.undelete_private_cloud(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*}:undelete' % client.transport._host, args[1])

def test_undelete_private_cloud_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.undelete_private_cloud(vmwareengine.UndeletePrivateCloudRequest(), name='name_value')

def test_undelete_private_cloud_rest_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListClustersRequest, dict])
def test_list_clusters_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_clusters(request)
    assert isinstance(response, pagers.ListClustersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_clusters_rest_required_fields(request_type=vmwareengine.ListClustersRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_clusters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_clusters._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListClustersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListClustersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_clusters(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_clusters_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_clusters._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_clusters_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_clusters') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_clusters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListClustersRequest.pb(vmwareengine.ListClustersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListClustersResponse.to_json(vmwareengine.ListClustersResponse())
        request = vmwareengine.ListClustersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListClustersResponse()
        client.list_clusters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_clusters_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListClustersRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_clusters(request)

def test_list_clusters_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListClustersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_clusters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/privateClouds/*}/clusters' % client.transport._host, args[1])

def test_list_clusters_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_clusters(vmwareengine.ListClustersRequest(), parent='parent_value')

def test_list_clusters_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()], next_page_token='abc'), vmwareengine.ListClustersResponse(clusters=[], next_page_token='def'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster()], next_page_token='ghi'), vmwareengine.ListClustersResponse(clusters=[vmwareengine_resources.Cluster(), vmwareengine_resources.Cluster()]))
        response = response + response
        response = tuple((vmwareengine.ListClustersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        pager = client.list_clusters(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.Cluster) for i in results))
        pages = list(client.list_clusters(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetClusterRequest, dict])
def test_get_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Cluster(name='name_value', state=vmwareengine_resources.Cluster.State.ACTIVE, management=True, uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Cluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_cluster(request)
    assert isinstance(response, vmwareengine_resources.Cluster)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.Cluster.State.ACTIVE
    assert response.management is True
    assert response.uid == 'uid_value'

def test_get_cluster_rest_required_fields(request_type=vmwareengine.GetClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.Cluster()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.Cluster.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_cluster_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_cluster') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetClusterRequest.pb(vmwareengine.GetClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.Cluster.to_json(vmwareengine_resources.Cluster())
        request = vmwareengine.GetClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.Cluster()
        client.get_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_cluster_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetClusterRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_cluster(request)

def test_get_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Cluster()
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Cluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*/clusters/*}' % client.transport._host, args[1])

def test_get_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_cluster(vmwareengine.GetClusterRequest(), name='name_value')

def test_get_cluster_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.CreateClusterRequest, dict])
def test_create_cluster_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request_init['cluster'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'state': 1, 'management': True, 'uid': 'uid_value', 'node_type_configs': {}}
    test_field = vmwareengine.CreateClusterRequest.meta.fields['cluster']

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
    for (field, value) in request_init['cluster'].items():
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
                for i in range(0, len(request_init['cluster'][field])):
                    del request_init['cluster'][field][i][subfield]
            else:
                del request_init['cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_create_cluster_rest_required_fields(request_type=vmwareengine.CreateClusterRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['cluster_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'clusterId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'clusterId' in jsonified_request
    assert jsonified_request['clusterId'] == request_init['cluster_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['clusterId'] = 'cluster_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('cluster_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'clusterId' in jsonified_request
    assert jsonified_request['clusterId'] == 'cluster_id_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_cluster(request)
            expected_params = [('clusterId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_cluster_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('clusterId', 'requestId', 'validateOnly')) & set(('parent', 'clusterId', 'cluster'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_cluster_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_create_cluster') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_create_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.CreateClusterRequest.pb(vmwareengine.CreateClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.CreateClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_cluster_rest_bad_request(transport: str='rest', request_type=vmwareengine.CreateClusterRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_cluster(request)

def test_create_cluster_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(parent='parent_value', cluster=vmwareengine_resources.Cluster(name='name_value'), cluster_id='cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/privateClouds/*}/clusters' % client.transport._host, args[1])

def test_create_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_cluster(vmwareengine.CreateClusterRequest(), parent='parent_value', cluster=vmwareengine_resources.Cluster(name='name_value'), cluster_id='cluster_id_value')

def test_create_cluster_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateClusterRequest, dict])
def test_update_cluster_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'cluster': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}}
    request_init['cluster'] = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'state': 1, 'management': True, 'uid': 'uid_value', 'node_type_configs': {}}
    test_field = vmwareengine.UpdateClusterRequest.meta.fields['cluster']

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
    for (field, value) in request_init['cluster'].items():
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
                for i in range(0, len(request_init['cluster'][field])):
                    del request_init['cluster'][field][i][subfield]
            else:
                del request_init['cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_update_cluster_rest_required_fields(request_type=vmwareengine.UpdateClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_cluster_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask', 'validateOnly')) & set(('updateMask', 'cluster'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_cluster_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_update_cluster') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_update_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UpdateClusterRequest.pb(vmwareengine.UpdateClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UpdateClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_cluster_rest_bad_request(transport: str='rest', request_type=vmwareengine.UpdateClusterRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'cluster': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_cluster(request)

def test_update_cluster_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'cluster': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}}
        mock_args = dict(cluster=vmwareengine_resources.Cluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{cluster.name=projects/*/locations/*/privateClouds/*/clusters/*}' % client.transport._host, args[1])

def test_update_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_cluster(vmwareengine.UpdateClusterRequest(), cluster=vmwareengine_resources.Cluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_cluster_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.DeleteClusterRequest, dict])
def test_delete_cluster_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_delete_cluster_rest_required_fields(request_type=vmwareengine.DeleteClusterRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_cluster_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_delete_cluster') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_delete_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.DeleteClusterRequest.pb(vmwareengine.DeleteClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.DeleteClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_cluster_rest_bad_request(transport: str='rest', request_type=vmwareengine.DeleteClusterRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_cluster(request)

def test_delete_cluster_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/clusters/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*/clusters/*}' % client.transport._host, args[1])

def test_delete_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_cluster(vmwareengine.DeleteClusterRequest(), name='name_value')

def test_delete_cluster_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListSubnetsRequest, dict])
def test_list_subnets_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListSubnetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListSubnetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_subnets(request)
    assert isinstance(response, pagers.ListSubnetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_subnets_rest_required_fields(request_type=vmwareengine.ListSubnetsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_subnets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_subnets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListSubnetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListSubnetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_subnets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_subnets_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_subnets._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_subnets_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_subnets') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_subnets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListSubnetsRequest.pb(vmwareengine.ListSubnetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListSubnetsResponse.to_json(vmwareengine.ListSubnetsResponse())
        request = vmwareengine.ListSubnetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListSubnetsResponse()
        client.list_subnets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_subnets_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListSubnetsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_subnets(request)

def test_list_subnets_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListSubnetsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListSubnetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_subnets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/privateClouds/*}/subnets' % client.transport._host, args[1])

def test_list_subnets_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_subnets(vmwareengine.ListSubnetsRequest(), parent='parent_value')

def test_list_subnets_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()], next_page_token='abc'), vmwareengine.ListSubnetsResponse(subnets=[], next_page_token='def'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet()], next_page_token='ghi'), vmwareengine.ListSubnetsResponse(subnets=[vmwareengine_resources.Subnet(), vmwareengine_resources.Subnet()]))
        response = response + response
        response = tuple((vmwareengine.ListSubnetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        pager = client.list_subnets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.Subnet) for i in results))
        pages = list(client.list_subnets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetSubnetRequest, dict])
def test_get_subnet_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Subnet(name='name_value', ip_cidr_range='ip_cidr_range_value', gateway_ip='gateway_ip_value', type_='type__value', state=vmwareengine_resources.Subnet.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Subnet.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_subnet(request)
    assert isinstance(response, vmwareengine_resources.Subnet)
    assert response.name == 'name_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.gateway_ip == 'gateway_ip_value'
    assert response.type_ == 'type__value'
    assert response.state == vmwareengine_resources.Subnet.State.ACTIVE

def test_get_subnet_rest_required_fields(request_type=vmwareengine.GetSubnetRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.Subnet()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.Subnet.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_subnet(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_subnet_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_subnet._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_subnet_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_subnet') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_subnet') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetSubnetRequest.pb(vmwareengine.GetSubnetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.Subnet.to_json(vmwareengine_resources.Subnet())
        request = vmwareengine.GetSubnetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.Subnet()
        client.get_subnet(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_subnet_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetSubnetRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_subnet(request)

def test_get_subnet_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Subnet()
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Subnet.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_subnet(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*/subnets/*}' % client.transport._host, args[1])

def test_get_subnet_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_subnet(vmwareengine.GetSubnetRequest(), name='name_value')

def test_get_subnet_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateSubnetRequest, dict])
def test_update_subnet_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'subnet': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4'}}
    request_init['subnet'] = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4', 'ip_cidr_range': 'ip_cidr_range_value', 'gateway_ip': 'gateway_ip_value', 'type_': 'type__value', 'state': 1}
    test_field = vmwareengine.UpdateSubnetRequest.meta.fields['subnet']

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
    for (field, value) in request_init['subnet'].items():
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
                for i in range(0, len(request_init['subnet'][field])):
                    del request_init['subnet'][field][i][subfield]
            else:
                del request_init['subnet'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_subnet(request)
    assert response.operation.name == 'operations/spam'

def test_update_subnet_rest_required_fields(request_type=vmwareengine.UpdateSubnetRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_subnet._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_subnet(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_subnet_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_subnet._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'subnet'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_subnet_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_update_subnet') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_update_subnet') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UpdateSubnetRequest.pb(vmwareengine.UpdateSubnetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UpdateSubnetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_subnet(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_subnet_rest_bad_request(transport: str='rest', request_type=vmwareengine.UpdateSubnetRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'subnet': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_subnet(request)

def test_update_subnet_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'subnet': {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/subnets/sample4'}}
        mock_args = dict(subnet=vmwareengine_resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_subnet(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{subnet.name=projects/*/locations/*/privateClouds/*/subnets/*}' % client.transport._host, args[1])

def test_update_subnet_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_subnet(vmwareengine.UpdateSubnetRequest(), subnet=vmwareengine_resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_subnet_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListNodeTypesRequest, dict])
def test_list_node_types_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListNodeTypesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListNodeTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_node_types(request)
    assert isinstance(response, pagers.ListNodeTypesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_node_types_rest_required_fields(request_type=vmwareengine.ListNodeTypesRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_node_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_node_types._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListNodeTypesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListNodeTypesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_node_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_node_types_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_node_types._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_node_types_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_node_types') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_node_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListNodeTypesRequest.pb(vmwareengine.ListNodeTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListNodeTypesResponse.to_json(vmwareengine.ListNodeTypesResponse())
        request = vmwareengine.ListNodeTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListNodeTypesResponse()
        client.list_node_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_node_types_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListNodeTypesRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_node_types(request)

def test_list_node_types_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListNodeTypesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListNodeTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_node_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/nodeTypes' % client.transport._host, args[1])

def test_list_node_types_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_node_types(vmwareengine.ListNodeTypesRequest(), parent='parent_value')

def test_list_node_types_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()], next_page_token='abc'), vmwareengine.ListNodeTypesResponse(node_types=[], next_page_token='def'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType()], next_page_token='ghi'), vmwareengine.ListNodeTypesResponse(node_types=[vmwareengine_resources.NodeType(), vmwareengine_resources.NodeType()]))
        response = response + response
        response = tuple((vmwareengine.ListNodeTypesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_node_types(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.NodeType) for i in results))
        pages = list(client.list_node_types(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetNodeTypeRequest, dict])
def test_get_node_type_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/nodeTypes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.NodeType(name='name_value', node_type_id='node_type_id_value', display_name='display_name_value', virtual_cpu_count=1846, total_core_count=1716, memory_gb=961, disk_size_gb=1261, available_custom_core_counts=[2974])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.NodeType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_node_type(request)
    assert isinstance(response, vmwareengine_resources.NodeType)
    assert response.name == 'name_value'
    assert response.node_type_id == 'node_type_id_value'
    assert response.display_name == 'display_name_value'
    assert response.virtual_cpu_count == 1846
    assert response.total_core_count == 1716
    assert response.memory_gb == 961
    assert response.disk_size_gb == 1261
    assert response.available_custom_core_counts == [2974]

def test_get_node_type_rest_required_fields(request_type=vmwareengine.GetNodeTypeRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_node_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_node_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.NodeType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.NodeType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_node_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_node_type_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_node_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_node_type_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_node_type') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_node_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetNodeTypeRequest.pb(vmwareengine.GetNodeTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.NodeType.to_json(vmwareengine_resources.NodeType())
        request = vmwareengine.GetNodeTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.NodeType()
        client.get_node_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_node_type_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetNodeTypeRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/nodeTypes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_node_type(request)

def test_get_node_type_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.NodeType()
        sample_request = {'name': 'projects/sample1/locations/sample2/nodeTypes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.NodeType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_node_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/nodeTypes/*}' % client.transport._host, args[1])

def test_get_node_type_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_node_type(vmwareengine.GetNodeTypeRequest(), name='name_value')

def test_get_node_type_rest_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ShowNsxCredentialsRequest, dict])
def test_show_nsx_credentials_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Credentials(username='username_value', password='password_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Credentials.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.show_nsx_credentials(request)
    assert isinstance(response, vmwareengine_resources.Credentials)
    assert response.username == 'username_value'
    assert response.password == 'password_value'

def test_show_nsx_credentials_rest_required_fields(request_type=vmwareengine.ShowNsxCredentialsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['private_cloud'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).show_nsx_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['privateCloud'] = 'private_cloud_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).show_nsx_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'privateCloud' in jsonified_request
    assert jsonified_request['privateCloud'] == 'private_cloud_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.Credentials()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.Credentials.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.show_nsx_credentials(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_show_nsx_credentials_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.show_nsx_credentials._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('privateCloud',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_show_nsx_credentials_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_show_nsx_credentials') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_show_nsx_credentials') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ShowNsxCredentialsRequest.pb(vmwareengine.ShowNsxCredentialsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.Credentials.to_json(vmwareengine_resources.Credentials())
        request = vmwareengine.ShowNsxCredentialsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.Credentials()
        client.show_nsx_credentials(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_show_nsx_credentials_rest_bad_request(transport: str='rest', request_type=vmwareengine.ShowNsxCredentialsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.show_nsx_credentials(request)

def test_show_nsx_credentials_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Credentials()
        sample_request = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(private_cloud='private_cloud_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Credentials.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.show_nsx_credentials(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{private_cloud=projects/*/locations/*/privateClouds/*}:showNsxCredentials' % client.transport._host, args[1])

def test_show_nsx_credentials_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.show_nsx_credentials(vmwareengine.ShowNsxCredentialsRequest(), private_cloud='private_cloud_value')

def test_show_nsx_credentials_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ShowVcenterCredentialsRequest, dict])
def test_show_vcenter_credentials_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Credentials(username='username_value', password='password_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Credentials.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.show_vcenter_credentials(request)
    assert isinstance(response, vmwareengine_resources.Credentials)
    assert response.username == 'username_value'
    assert response.password == 'password_value'

def test_show_vcenter_credentials_rest_required_fields(request_type=vmwareengine.ShowVcenterCredentialsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['private_cloud'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).show_vcenter_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['privateCloud'] = 'private_cloud_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).show_vcenter_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'privateCloud' in jsonified_request
    assert jsonified_request['privateCloud'] == 'private_cloud_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.Credentials()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.Credentials.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.show_vcenter_credentials(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_show_vcenter_credentials_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.show_vcenter_credentials._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('privateCloud',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_show_vcenter_credentials_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_show_vcenter_credentials') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_show_vcenter_credentials') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ShowVcenterCredentialsRequest.pb(vmwareengine.ShowVcenterCredentialsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.Credentials.to_json(vmwareengine_resources.Credentials())
        request = vmwareengine.ShowVcenterCredentialsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.Credentials()
        client.show_vcenter_credentials(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_show_vcenter_credentials_rest_bad_request(transport: str='rest', request_type=vmwareengine.ShowVcenterCredentialsRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.show_vcenter_credentials(request)

def test_show_vcenter_credentials_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.Credentials()
        sample_request = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(private_cloud='private_cloud_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.Credentials.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.show_vcenter_credentials(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{private_cloud=projects/*/locations/*/privateClouds/*}:showVcenterCredentials' % client.transport._host, args[1])

def test_show_vcenter_credentials_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.show_vcenter_credentials(vmwareengine.ShowVcenterCredentialsRequest(), private_cloud='private_cloud_value')

def test_show_vcenter_credentials_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ResetNsxCredentialsRequest, dict])
def test_reset_nsx_credentials_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.reset_nsx_credentials(request)
    assert response.operation.name == 'operations/spam'

def test_reset_nsx_credentials_rest_required_fields(request_type=vmwareengine.ResetNsxCredentialsRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['private_cloud'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reset_nsx_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['privateCloud'] = 'private_cloud_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reset_nsx_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'privateCloud' in jsonified_request
    assert jsonified_request['privateCloud'] == 'private_cloud_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.reset_nsx_credentials(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_reset_nsx_credentials_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reset_nsx_credentials._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('privateCloud',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reset_nsx_credentials_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_reset_nsx_credentials') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_reset_nsx_credentials') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ResetNsxCredentialsRequest.pb(vmwareengine.ResetNsxCredentialsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.ResetNsxCredentialsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.reset_nsx_credentials(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reset_nsx_credentials_rest_bad_request(transport: str='rest', request_type=vmwareengine.ResetNsxCredentialsRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reset_nsx_credentials(request)

def test_reset_nsx_credentials_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(private_cloud='private_cloud_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.reset_nsx_credentials(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{private_cloud=projects/*/locations/*/privateClouds/*}:resetNsxCredentials' % client.transport._host, args[1])

def test_reset_nsx_credentials_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.reset_nsx_credentials(vmwareengine.ResetNsxCredentialsRequest(), private_cloud='private_cloud_value')

def test_reset_nsx_credentials_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ResetVcenterCredentialsRequest, dict])
def test_reset_vcenter_credentials_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.reset_vcenter_credentials(request)
    assert response.operation.name == 'operations/spam'

def test_reset_vcenter_credentials_rest_required_fields(request_type=vmwareengine.ResetVcenterCredentialsRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['private_cloud'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reset_vcenter_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['privateCloud'] = 'private_cloud_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reset_vcenter_credentials._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'privateCloud' in jsonified_request
    assert jsonified_request['privateCloud'] == 'private_cloud_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.reset_vcenter_credentials(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_reset_vcenter_credentials_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reset_vcenter_credentials._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('privateCloud',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reset_vcenter_credentials_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_reset_vcenter_credentials') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_reset_vcenter_credentials') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ResetVcenterCredentialsRequest.pb(vmwareengine.ResetVcenterCredentialsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.ResetVcenterCredentialsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.reset_vcenter_credentials(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reset_vcenter_credentials_rest_bad_request(transport: str='rest', request_type=vmwareengine.ResetVcenterCredentialsRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reset_vcenter_credentials(request)

def test_reset_vcenter_credentials_rest_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'private_cloud': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(private_cloud='private_cloud_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.reset_vcenter_credentials(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{private_cloud=projects/*/locations/*/privateClouds/*}:resetVcenterCredentials' % client.transport._host, args[1])

def test_reset_vcenter_credentials_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.reset_vcenter_credentials(vmwareengine.ResetVcenterCredentialsRequest(), private_cloud='private_cloud_value')

def test_reset_vcenter_credentials_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.CreateHcxActivationKeyRequest, dict])
def test_create_hcx_activation_key_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request_init['hcx_activation_key'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'activation_key': 'activation_key_value', 'uid': 'uid_value'}
    test_field = vmwareengine.CreateHcxActivationKeyRequest.meta.fields['hcx_activation_key']

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
    for (field, value) in request_init['hcx_activation_key'].items():
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
                for i in range(0, len(request_init['hcx_activation_key'][field])):
                    del request_init['hcx_activation_key'][field][i][subfield]
            else:
                del request_init['hcx_activation_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_hcx_activation_key(request)
    assert response.operation.name == 'operations/spam'

def test_create_hcx_activation_key_rest_required_fields(request_type=vmwareengine.CreateHcxActivationKeyRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['hcx_activation_key_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'hcxActivationKeyId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_hcx_activation_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'hcxActivationKeyId' in jsonified_request
    assert jsonified_request['hcxActivationKeyId'] == request_init['hcx_activation_key_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['hcxActivationKeyId'] = 'hcx_activation_key_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_hcx_activation_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('hcx_activation_key_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'hcxActivationKeyId' in jsonified_request
    assert jsonified_request['hcxActivationKeyId'] == 'hcx_activation_key_id_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_hcx_activation_key(request)
            expected_params = [('hcxActivationKeyId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_hcx_activation_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_hcx_activation_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('hcxActivationKeyId', 'requestId')) & set(('parent', 'hcxActivationKey', 'hcxActivationKeyId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_hcx_activation_key_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_create_hcx_activation_key') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_create_hcx_activation_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.CreateHcxActivationKeyRequest.pb(vmwareengine.CreateHcxActivationKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.CreateHcxActivationKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_hcx_activation_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_hcx_activation_key_rest_bad_request(transport: str='rest', request_type=vmwareengine.CreateHcxActivationKeyRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_hcx_activation_key(request)

def test_create_hcx_activation_key_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(parent='parent_value', hcx_activation_key=vmwareengine_resources.HcxActivationKey(name='name_value'), hcx_activation_key_id='hcx_activation_key_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_hcx_activation_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/privateClouds/*}/hcxActivationKeys' % client.transport._host, args[1])

def test_create_hcx_activation_key_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_hcx_activation_key(vmwareengine.CreateHcxActivationKeyRequest(), parent='parent_value', hcx_activation_key=vmwareengine_resources.HcxActivationKey(name='name_value'), hcx_activation_key_id='hcx_activation_key_id_value')

def test_create_hcx_activation_key_rest_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListHcxActivationKeysRequest, dict])
def test_list_hcx_activation_keys_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListHcxActivationKeysResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListHcxActivationKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_hcx_activation_keys(request)
    assert isinstance(response, pagers.ListHcxActivationKeysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_hcx_activation_keys_rest_required_fields(request_type=vmwareengine.ListHcxActivationKeysRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_hcx_activation_keys._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_hcx_activation_keys._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListHcxActivationKeysResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListHcxActivationKeysResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_hcx_activation_keys(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_hcx_activation_keys_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_hcx_activation_keys._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_hcx_activation_keys_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_hcx_activation_keys') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_hcx_activation_keys') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListHcxActivationKeysRequest.pb(vmwareengine.ListHcxActivationKeysRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListHcxActivationKeysResponse.to_json(vmwareengine.ListHcxActivationKeysResponse())
        request = vmwareengine.ListHcxActivationKeysRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListHcxActivationKeysResponse()
        client.list_hcx_activation_keys(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_hcx_activation_keys_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListHcxActivationKeysRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_hcx_activation_keys(request)

def test_list_hcx_activation_keys_rest_flattened():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListHcxActivationKeysResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListHcxActivationKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_hcx_activation_keys(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/privateClouds/*}/hcxActivationKeys' % client.transport._host, args[1])

def test_list_hcx_activation_keys_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_hcx_activation_keys(vmwareengine.ListHcxActivationKeysRequest(), parent='parent_value')

def test_list_hcx_activation_keys_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()], next_page_token='abc'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[], next_page_token='def'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey()], next_page_token='ghi'), vmwareengine.ListHcxActivationKeysResponse(hcx_activation_keys=[vmwareengine_resources.HcxActivationKey(), vmwareengine_resources.HcxActivationKey()]))
        response = response + response
        response = tuple((vmwareengine.ListHcxActivationKeysResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateClouds/sample3'}
        pager = client.list_hcx_activation_keys(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.HcxActivationKey) for i in results))
        pages = list(client.list_hcx_activation_keys(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.GetHcxActivationKeyRequest, dict])
def test_get_hcx_activation_key_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/hcxActivationKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.HcxActivationKey(name='name_value', state=vmwareengine_resources.HcxActivationKey.State.AVAILABLE, activation_key='activation_key_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.HcxActivationKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_hcx_activation_key(request)
    assert isinstance(response, vmwareengine_resources.HcxActivationKey)
    assert response.name == 'name_value'
    assert response.state == vmwareengine_resources.HcxActivationKey.State.AVAILABLE
    assert response.activation_key == 'activation_key_value'
    assert response.uid == 'uid_value'

def test_get_hcx_activation_key_rest_required_fields(request_type=vmwareengine.GetHcxActivationKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_hcx_activation_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_hcx_activation_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.HcxActivationKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.HcxActivationKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_hcx_activation_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_hcx_activation_key_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_hcx_activation_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_hcx_activation_key_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_hcx_activation_key') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_hcx_activation_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetHcxActivationKeyRequest.pb(vmwareengine.GetHcxActivationKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.HcxActivationKey.to_json(vmwareengine_resources.HcxActivationKey())
        request = vmwareengine.GetHcxActivationKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.HcxActivationKey()
        client.get_hcx_activation_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_hcx_activation_key_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetHcxActivationKeyRequest):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/hcxActivationKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_hcx_activation_key(request)

def test_get_hcx_activation_key_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.HcxActivationKey()
        sample_request = {'name': 'projects/sample1/locations/sample2/privateClouds/sample3/hcxActivationKeys/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.HcxActivationKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_hcx_activation_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateClouds/*/hcxActivationKeys/*}' % client.transport._host, args[1])

def test_get_hcx_activation_key_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_hcx_activation_key(vmwareengine.GetHcxActivationKeyRequest(), name='name_value')

def test_get_hcx_activation_key_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.GetNetworkPolicyRequest, dict])
def test_get_network_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.NetworkPolicy(name='name_value', edge_services_cidr='edge_services_cidr_value', uid='uid_value', vmware_engine_network='vmware_engine_network_value', description='description_value', vmware_engine_network_canonical='vmware_engine_network_canonical_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.NetworkPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_network_policy(request)
    assert isinstance(response, vmwareengine_resources.NetworkPolicy)
    assert response.name == 'name_value'
    assert response.edge_services_cidr == 'edge_services_cidr_value'
    assert response.uid == 'uid_value'
    assert response.vmware_engine_network == 'vmware_engine_network_value'
    assert response.description == 'description_value'
    assert response.vmware_engine_network_canonical == 'vmware_engine_network_canonical_value'

def test_get_network_policy_rest_required_fields(request_type=vmwareengine.GetNetworkPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_network_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_network_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.NetworkPolicy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.NetworkPolicy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_network_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_network_policy_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_network_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_network_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_network_policy') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_network_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetNetworkPolicyRequest.pb(vmwareengine.GetNetworkPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.NetworkPolicy.to_json(vmwareengine_resources.NetworkPolicy())
        request = vmwareengine.GetNetworkPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.NetworkPolicy()
        client.get_network_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_network_policy_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetNetworkPolicyRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_network_policy(request)

def test_get_network_policy_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.NetworkPolicy()
        sample_request = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.NetworkPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_network_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/networkPolicies/*}' % client.transport._host, args[1])

def test_get_network_policy_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_network_policy(vmwareengine.GetNetworkPolicyRequest(), name='name_value')

def test_get_network_policy_rest_error():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListNetworkPoliciesRequest, dict])
def test_list_network_policies_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListNetworkPoliciesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListNetworkPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_network_policies(request)
    assert isinstance(response, pagers.ListNetworkPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_network_policies_rest_required_fields(request_type=vmwareengine.ListNetworkPoliciesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_network_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_network_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListNetworkPoliciesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListNetworkPoliciesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_network_policies(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_network_policies_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_network_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_network_policies_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_network_policies') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_network_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListNetworkPoliciesRequest.pb(vmwareengine.ListNetworkPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListNetworkPoliciesResponse.to_json(vmwareengine.ListNetworkPoliciesResponse())
        request = vmwareengine.ListNetworkPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListNetworkPoliciesResponse()
        client.list_network_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_network_policies_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListNetworkPoliciesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_network_policies(request)

def test_list_network_policies_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListNetworkPoliciesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListNetworkPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_network_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/networkPolicies' % client.transport._host, args[1])

def test_list_network_policies_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_network_policies(vmwareengine.ListNetworkPoliciesRequest(), parent='parent_value')

def test_list_network_policies_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()], next_page_token='abc'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[], next_page_token='def'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy()], next_page_token='ghi'), vmwareengine.ListNetworkPoliciesResponse(network_policies=[vmwareengine_resources.NetworkPolicy(), vmwareengine_resources.NetworkPolicy()]))
        response = response + response
        response = tuple((vmwareengine.ListNetworkPoliciesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_network_policies(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.NetworkPolicy) for i in results))
        pages = list(client.list_network_policies(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.CreateNetworkPolicyRequest, dict])
def test_create_network_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['network_policy'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'internet_access': {'enabled': True, 'state': 1}, 'external_ip': {}, 'edge_services_cidr': 'edge_services_cidr_value', 'uid': 'uid_value', 'vmware_engine_network': 'vmware_engine_network_value', 'description': 'description_value', 'vmware_engine_network_canonical': 'vmware_engine_network_canonical_value'}
    test_field = vmwareengine.CreateNetworkPolicyRequest.meta.fields['network_policy']

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
    for (field, value) in request_init['network_policy'].items():
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
                for i in range(0, len(request_init['network_policy'][field])):
                    del request_init['network_policy'][field][i][subfield]
            else:
                del request_init['network_policy'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_network_policy(request)
    assert response.operation.name == 'operations/spam'

def test_create_network_policy_rest_required_fields(request_type=vmwareengine.CreateNetworkPolicyRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['network_policy_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'networkPolicyId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_network_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'networkPolicyId' in jsonified_request
    assert jsonified_request['networkPolicyId'] == request_init['network_policy_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['networkPolicyId'] = 'network_policy_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_network_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('network_policy_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'networkPolicyId' in jsonified_request
    assert jsonified_request['networkPolicyId'] == 'network_policy_id_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_network_policy(request)
            expected_params = [('networkPolicyId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_network_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_network_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('networkPolicyId', 'requestId')) & set(('parent', 'networkPolicyId', 'networkPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_network_policy_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_create_network_policy') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_create_network_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.CreateNetworkPolicyRequest.pb(vmwareengine.CreateNetworkPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.CreateNetworkPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_network_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_network_policy_rest_bad_request(transport: str='rest', request_type=vmwareengine.CreateNetworkPolicyRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_network_policy(request)

def test_create_network_policy_rest_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), network_policy_id='network_policy_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_network_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/networkPolicies' % client.transport._host, args[1])

def test_create_network_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_network_policy(vmwareengine.CreateNetworkPolicyRequest(), parent='parent_value', network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), network_policy_id='network_policy_id_value')

def test_create_network_policy_rest_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateNetworkPolicyRequest, dict])
def test_update_network_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'network_policy': {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}}
    request_init['network_policy'] = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'internet_access': {'enabled': True, 'state': 1}, 'external_ip': {}, 'edge_services_cidr': 'edge_services_cidr_value', 'uid': 'uid_value', 'vmware_engine_network': 'vmware_engine_network_value', 'description': 'description_value', 'vmware_engine_network_canonical': 'vmware_engine_network_canonical_value'}
    test_field = vmwareengine.UpdateNetworkPolicyRequest.meta.fields['network_policy']

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
    for (field, value) in request_init['network_policy'].items():
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
                for i in range(0, len(request_init['network_policy'][field])):
                    del request_init['network_policy'][field][i][subfield]
            else:
                del request_init['network_policy'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_network_policy(request)
    assert response.operation.name == 'operations/spam'

def test_update_network_policy_rest_required_fields(request_type=vmwareengine.UpdateNetworkPolicyRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_network_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_network_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_network_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_network_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_network_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('networkPolicy', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_network_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_update_network_policy') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_update_network_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UpdateNetworkPolicyRequest.pb(vmwareengine.UpdateNetworkPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UpdateNetworkPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_network_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_network_policy_rest_bad_request(transport: str='rest', request_type=vmwareengine.UpdateNetworkPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'network_policy': {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_network_policy(request)

def test_update_network_policy_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'network_policy': {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}}
        mock_args = dict(network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_network_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{network_policy.name=projects/*/locations/*/networkPolicies/*}' % client.transport._host, args[1])

def test_update_network_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_network_policy(vmwareengine.UpdateNetworkPolicyRequest(), network_policy=vmwareengine_resources.NetworkPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_network_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.DeleteNetworkPolicyRequest, dict])
def test_delete_network_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_network_policy(request)
    assert response.operation.name == 'operations/spam'

def test_delete_network_policy_rest_required_fields(request_type=vmwareengine.DeleteNetworkPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_network_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_network_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_network_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_network_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_network_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_network_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_delete_network_policy') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_delete_network_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.DeleteNetworkPolicyRequest.pb(vmwareengine.DeleteNetworkPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.DeleteNetworkPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_network_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_network_policy_rest_bad_request(transport: str='rest', request_type=vmwareengine.DeleteNetworkPolicyRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_network_policy(request)

def test_delete_network_policy_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/networkPolicies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_network_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/networkPolicies/*}' % client.transport._host, args[1])

def test_delete_network_policy_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_network_policy(vmwareengine.DeleteNetworkPolicyRequest(), name='name_value')

def test_delete_network_policy_rest_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.CreateVmwareEngineNetworkRequest, dict])
def test_create_vmware_engine_network_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['vmware_engine_network'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'vpc_networks': [{'type_': 1, 'network': 'network_value'}], 'state': 1, 'type_': 1, 'uid': 'uid_value', 'etag': 'etag_value'}
    test_field = vmwareengine.CreateVmwareEngineNetworkRequest.meta.fields['vmware_engine_network']

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
    for (field, value) in request_init['vmware_engine_network'].items():
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
                for i in range(0, len(request_init['vmware_engine_network'][field])):
                    del request_init['vmware_engine_network'][field][i][subfield]
            else:
                del request_init['vmware_engine_network'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_vmware_engine_network(request)
    assert response.operation.name == 'operations/spam'

def test_create_vmware_engine_network_rest_required_fields(request_type=vmwareengine.CreateVmwareEngineNetworkRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['vmware_engine_network_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'vmwareEngineNetworkId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_vmware_engine_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'vmwareEngineNetworkId' in jsonified_request
    assert jsonified_request['vmwareEngineNetworkId'] == request_init['vmware_engine_network_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['vmwareEngineNetworkId'] = 'vmware_engine_network_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_vmware_engine_network._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'vmware_engine_network_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'vmwareEngineNetworkId' in jsonified_request
    assert jsonified_request['vmwareEngineNetworkId'] == 'vmware_engine_network_id_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_vmware_engine_network(request)
            expected_params = [('vmwareEngineNetworkId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_vmware_engine_network_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_vmware_engine_network._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'vmwareEngineNetworkId')) & set(('parent', 'vmwareEngineNetworkId', 'vmwareEngineNetwork'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_vmware_engine_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_create_vmware_engine_network') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_create_vmware_engine_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.CreateVmwareEngineNetworkRequest.pb(vmwareengine.CreateVmwareEngineNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.CreateVmwareEngineNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_vmware_engine_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_vmware_engine_network_rest_bad_request(transport: str='rest', request_type=vmwareengine.CreateVmwareEngineNetworkRequest):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_vmware_engine_network(request)

def test_create_vmware_engine_network_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), vmware_engine_network_id='vmware_engine_network_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_vmware_engine_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/vmwareEngineNetworks' % client.transport._host, args[1])

def test_create_vmware_engine_network_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_vmware_engine_network(vmwareengine.CreateVmwareEngineNetworkRequest(), parent='parent_value', vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), vmware_engine_network_id='vmware_engine_network_id_value')

def test_create_vmware_engine_network_rest_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.UpdateVmwareEngineNetworkRequest, dict])
def test_update_vmware_engine_network_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'vmware_engine_network': {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}}
    request_init['vmware_engine_network'] = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'vpc_networks': [{'type_': 1, 'network': 'network_value'}], 'state': 1, 'type_': 1, 'uid': 'uid_value', 'etag': 'etag_value'}
    test_field = vmwareengine.UpdateVmwareEngineNetworkRequest.meta.fields['vmware_engine_network']

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
    for (field, value) in request_init['vmware_engine_network'].items():
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
                for i in range(0, len(request_init['vmware_engine_network'][field])):
                    del request_init['vmware_engine_network'][field][i][subfield]
            else:
                del request_init['vmware_engine_network'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_vmware_engine_network(request)
    assert response.operation.name == 'operations/spam'

def test_update_vmware_engine_network_rest_required_fields(request_type=vmwareengine.UpdateVmwareEngineNetworkRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_vmware_engine_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_vmware_engine_network._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_vmware_engine_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_vmware_engine_network_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_vmware_engine_network._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('vmwareEngineNetwork', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_vmware_engine_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_update_vmware_engine_network') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_update_vmware_engine_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UpdateVmwareEngineNetworkRequest.pb(vmwareengine.UpdateVmwareEngineNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UpdateVmwareEngineNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_vmware_engine_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_vmware_engine_network_rest_bad_request(transport: str='rest', request_type=vmwareengine.UpdateVmwareEngineNetworkRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'vmware_engine_network': {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_vmware_engine_network(request)

def test_update_vmware_engine_network_rest_flattened():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'vmware_engine_network': {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}}
        mock_args = dict(vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_vmware_engine_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{vmware_engine_network.name=projects/*/locations/*/vmwareEngineNetworks/*}' % client.transport._host, args[1])

def test_update_vmware_engine_network_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_vmware_engine_network(vmwareengine.UpdateVmwareEngineNetworkRequest(), vmware_engine_network=vmwareengine_resources.VmwareEngineNetwork(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_vmware_engine_network_rest_error():
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.DeleteVmwareEngineNetworkRequest, dict])
def test_delete_vmware_engine_network_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_vmware_engine_network(request)
    assert response.operation.name == 'operations/spam'

def test_delete_vmware_engine_network_rest_required_fields(request_type=vmwareengine.DeleteVmwareEngineNetworkRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_vmware_engine_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_vmware_engine_network._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_vmware_engine_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_vmware_engine_network_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_vmware_engine_network._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'requestId')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_vmware_engine_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_delete_vmware_engine_network') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_delete_vmware_engine_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.DeleteVmwareEngineNetworkRequest.pb(vmwareengine.DeleteVmwareEngineNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.DeleteVmwareEngineNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_vmware_engine_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_vmware_engine_network_rest_bad_request(transport: str='rest', request_type=vmwareengine.DeleteVmwareEngineNetworkRequest):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_vmware_engine_network(request)

def test_delete_vmware_engine_network_rest_flattened():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_vmware_engine_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/vmwareEngineNetworks/*}' % client.transport._host, args[1])

def test_delete_vmware_engine_network_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_vmware_engine_network(vmwareengine.DeleteVmwareEngineNetworkRequest(), name='name_value')

def test_delete_vmware_engine_network_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.GetVmwareEngineNetworkRequest, dict])
def test_get_vmware_engine_network_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.VmwareEngineNetwork(name='name_value', description='description_value', state=vmwareengine_resources.VmwareEngineNetwork.State.CREATING, type_=vmwareengine_resources.VmwareEngineNetwork.Type.LEGACY, uid='uid_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.VmwareEngineNetwork.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_vmware_engine_network(request)
    assert isinstance(response, vmwareengine_resources.VmwareEngineNetwork)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == vmwareengine_resources.VmwareEngineNetwork.State.CREATING
    assert response.type_ == vmwareengine_resources.VmwareEngineNetwork.Type.LEGACY
    assert response.uid == 'uid_value'
    assert response.etag == 'etag_value'

def test_get_vmware_engine_network_rest_required_fields(request_type=vmwareengine.GetVmwareEngineNetworkRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_vmware_engine_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_vmware_engine_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.VmwareEngineNetwork()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.VmwareEngineNetwork.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_vmware_engine_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_vmware_engine_network_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_vmware_engine_network._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_vmware_engine_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_vmware_engine_network') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_vmware_engine_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetVmwareEngineNetworkRequest.pb(vmwareengine.GetVmwareEngineNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.VmwareEngineNetwork.to_json(vmwareengine_resources.VmwareEngineNetwork())
        request = vmwareengine.GetVmwareEngineNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.VmwareEngineNetwork()
        client.get_vmware_engine_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_vmware_engine_network_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetVmwareEngineNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_vmware_engine_network(request)

def test_get_vmware_engine_network_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.VmwareEngineNetwork()
        sample_request = {'name': 'projects/sample1/locations/sample2/vmwareEngineNetworks/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.VmwareEngineNetwork.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_vmware_engine_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/vmwareEngineNetworks/*}' % client.transport._host, args[1])

def test_get_vmware_engine_network_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_vmware_engine_network(vmwareengine.GetVmwareEngineNetworkRequest(), name='name_value')

def test_get_vmware_engine_network_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListVmwareEngineNetworksRequest, dict])
def test_list_vmware_engine_networks_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListVmwareEngineNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListVmwareEngineNetworksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_vmware_engine_networks(request)
    assert isinstance(response, pagers.ListVmwareEngineNetworksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_vmware_engine_networks_rest_required_fields(request_type=vmwareengine.ListVmwareEngineNetworksRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_vmware_engine_networks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_vmware_engine_networks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListVmwareEngineNetworksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListVmwareEngineNetworksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_vmware_engine_networks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_vmware_engine_networks_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_vmware_engine_networks._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_vmware_engine_networks_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_vmware_engine_networks') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_vmware_engine_networks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListVmwareEngineNetworksRequest.pb(vmwareengine.ListVmwareEngineNetworksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListVmwareEngineNetworksResponse.to_json(vmwareengine.ListVmwareEngineNetworksResponse())
        request = vmwareengine.ListVmwareEngineNetworksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListVmwareEngineNetworksResponse()
        client.list_vmware_engine_networks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_vmware_engine_networks_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListVmwareEngineNetworksRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_vmware_engine_networks(request)

def test_list_vmware_engine_networks_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListVmwareEngineNetworksResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListVmwareEngineNetworksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_vmware_engine_networks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/vmwareEngineNetworks' % client.transport._host, args[1])

def test_list_vmware_engine_networks_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_vmware_engine_networks(vmwareengine.ListVmwareEngineNetworksRequest(), parent='parent_value')

def test_list_vmware_engine_networks_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()], next_page_token='abc'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[], next_page_token='def'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork()], next_page_token='ghi'), vmwareengine.ListVmwareEngineNetworksResponse(vmware_engine_networks=[vmwareengine_resources.VmwareEngineNetwork(), vmwareengine_resources.VmwareEngineNetwork()]))
        response = response + response
        response = tuple((vmwareengine.ListVmwareEngineNetworksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_vmware_engine_networks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.VmwareEngineNetwork) for i in results))
        pages = list(client.list_vmware_engine_networks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.CreatePrivateConnectionRequest, dict])
def test_create_private_connection_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['private_connection'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'state': 1, 'vmware_engine_network': 'vmware_engine_network_value', 'vmware_engine_network_canonical': 'vmware_engine_network_canonical_value', 'type_': 1, 'peering_id': 'peering_id_value', 'routing_mode': 1, 'uid': 'uid_value', 'service_network': 'service_network_value', 'peering_state': 1}
    test_field = vmwareengine.CreatePrivateConnectionRequest.meta.fields['private_connection']

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
    for (field, value) in request_init['private_connection'].items():
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
                for i in range(0, len(request_init['private_connection'][field])):
                    del request_init['private_connection'][field][i][subfield]
            else:
                del request_init['private_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_private_connection(request)
    assert response.operation.name == 'operations/spam'

def test_create_private_connection_rest_required_fields(request_type=vmwareengine.CreatePrivateConnectionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['private_connection_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'privateConnectionId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_private_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'privateConnectionId' in jsonified_request
    assert jsonified_request['privateConnectionId'] == request_init['private_connection_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['privateConnectionId'] = 'private_connection_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_private_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('private_connection_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'privateConnectionId' in jsonified_request
    assert jsonified_request['privateConnectionId'] == 'private_connection_id_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_private_connection(request)
            expected_params = [('privateConnectionId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_private_connection_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_private_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('privateConnectionId', 'requestId')) & set(('parent', 'privateConnectionId', 'privateConnection'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_private_connection_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_create_private_connection') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_create_private_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.CreatePrivateConnectionRequest.pb(vmwareengine.CreatePrivateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.CreatePrivateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_private_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_private_connection_rest_bad_request(transport: str='rest', request_type=vmwareengine.CreatePrivateConnectionRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_private_connection(request)

def test_create_private_connection_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), private_connection_id='private_connection_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_private_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/privateConnections' % client.transport._host, args[1])

def test_create_private_connection_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_private_connection(vmwareengine.CreatePrivateConnectionRequest(), parent='parent_value', private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), private_connection_id='private_connection_id_value')

def test_create_private_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.GetPrivateConnectionRequest, dict])
def test_get_private_connection_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.PrivateConnection(name='name_value', description='description_value', state=vmwareengine_resources.PrivateConnection.State.CREATING, vmware_engine_network='vmware_engine_network_value', vmware_engine_network_canonical='vmware_engine_network_canonical_value', type_=vmwareengine_resources.PrivateConnection.Type.PRIVATE_SERVICE_ACCESS, peering_id='peering_id_value', routing_mode=vmwareengine_resources.PrivateConnection.RoutingMode.GLOBAL, uid='uid_value', service_network='service_network_value', peering_state=vmwareengine_resources.PrivateConnection.PeeringState.PEERING_ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.PrivateConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_private_connection(request)
    assert isinstance(response, vmwareengine_resources.PrivateConnection)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == vmwareengine_resources.PrivateConnection.State.CREATING
    assert response.vmware_engine_network == 'vmware_engine_network_value'
    assert response.vmware_engine_network_canonical == 'vmware_engine_network_canonical_value'
    assert response.type_ == vmwareengine_resources.PrivateConnection.Type.PRIVATE_SERVICE_ACCESS
    assert response.peering_id == 'peering_id_value'
    assert response.routing_mode == vmwareengine_resources.PrivateConnection.RoutingMode.GLOBAL
    assert response.uid == 'uid_value'
    assert response.service_network == 'service_network_value'
    assert response.peering_state == vmwareengine_resources.PrivateConnection.PeeringState.PEERING_ACTIVE

def test_get_private_connection_rest_required_fields(request_type=vmwareengine.GetPrivateConnectionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_private_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_private_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine_resources.PrivateConnection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine_resources.PrivateConnection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_private_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_private_connection_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_private_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_private_connection_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_get_private_connection') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_get_private_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.GetPrivateConnectionRequest.pb(vmwareengine.GetPrivateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine_resources.PrivateConnection.to_json(vmwareengine_resources.PrivateConnection())
        request = vmwareengine.GetPrivateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine_resources.PrivateConnection()
        client.get_private_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_private_connection_rest_bad_request(transport: str='rest', request_type=vmwareengine.GetPrivateConnectionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_private_connection(request)

def test_get_private_connection_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine_resources.PrivateConnection()
        sample_request = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine_resources.PrivateConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_private_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateConnections/*}' % client.transport._host, args[1])

def test_get_private_connection_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_private_connection(vmwareengine.GetPrivateConnectionRequest(), name='name_value')

def test_get_private_connection_rest_error():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListPrivateConnectionsRequest, dict])
def test_list_private_connections_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListPrivateConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListPrivateConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_private_connections(request)
    assert isinstance(response, pagers.ListPrivateConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_private_connections_rest_required_fields(request_type=vmwareengine.ListPrivateConnectionsRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_private_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_private_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListPrivateConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListPrivateConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_private_connections(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_private_connections_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_private_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_private_connections_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_private_connections') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_private_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListPrivateConnectionsRequest.pb(vmwareengine.ListPrivateConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListPrivateConnectionsResponse.to_json(vmwareengine.ListPrivateConnectionsResponse())
        request = vmwareengine.ListPrivateConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListPrivateConnectionsResponse()
        client.list_private_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_private_connections_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListPrivateConnectionsRequest):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_private_connections(request)

def test_list_private_connections_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListPrivateConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListPrivateConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_private_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/privateConnections' % client.transport._host, args[1])

def test_list_private_connections_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_private_connections(vmwareengine.ListPrivateConnectionsRequest(), parent='parent_value')

def test_list_private_connections_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()], next_page_token='abc'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[], next_page_token='def'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionsResponse(private_connections=[vmwareengine_resources.PrivateConnection(), vmwareengine_resources.PrivateConnection()]))
        response = response + response
        response = tuple((vmwareengine.ListPrivateConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_private_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.PrivateConnection) for i in results))
        pages = list(client.list_private_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vmwareengine.UpdatePrivateConnectionRequest, dict])
def test_update_private_connection_rest(request_type):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'private_connection': {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}}
    request_init['private_connection'] = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'state': 1, 'vmware_engine_network': 'vmware_engine_network_value', 'vmware_engine_network_canonical': 'vmware_engine_network_canonical_value', 'type_': 1, 'peering_id': 'peering_id_value', 'routing_mode': 1, 'uid': 'uid_value', 'service_network': 'service_network_value', 'peering_state': 1}
    test_field = vmwareengine.UpdatePrivateConnectionRequest.meta.fields['private_connection']

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
    for (field, value) in request_init['private_connection'].items():
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
                for i in range(0, len(request_init['private_connection'][field])):
                    del request_init['private_connection'][field][i][subfield]
            else:
                del request_init['private_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_private_connection(request)
    assert response.operation.name == 'operations/spam'

def test_update_private_connection_rest_required_fields(request_type=vmwareengine.UpdatePrivateConnectionRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_private_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_private_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_private_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_private_connection_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_private_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('privateConnection', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_private_connection_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_update_private_connection') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_update_private_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.UpdatePrivateConnectionRequest.pb(vmwareengine.UpdatePrivateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.UpdatePrivateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_private_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_private_connection_rest_bad_request(transport: str='rest', request_type=vmwareengine.UpdatePrivateConnectionRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'private_connection': {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_private_connection(request)

def test_update_private_connection_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'private_connection': {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}}
        mock_args = dict(private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_private_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{private_connection.name=projects/*/locations/*/privateConnections/*}' % client.transport._host, args[1])

def test_update_private_connection_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_private_connection(vmwareengine.UpdatePrivateConnectionRequest(), private_connection=vmwareengine_resources.PrivateConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_private_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.DeletePrivateConnectionRequest, dict])
def test_delete_private_connection_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_private_connection(request)
    assert response.operation.name == 'operations/spam'

def test_delete_private_connection_rest_required_fields(request_type=vmwareengine.DeletePrivateConnectionRequest):
    if False:
        return 10
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_private_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_private_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_private_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_private_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_private_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_private_connection_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_delete_private_connection') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_delete_private_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.DeletePrivateConnectionRequest.pb(vmwareengine.DeletePrivateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vmwareengine.DeletePrivateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_private_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_private_connection_rest_bad_request(transport: str='rest', request_type=vmwareengine.DeletePrivateConnectionRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_private_connection(request)

def test_delete_private_connection_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/privateConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_private_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/privateConnections/*}' % client.transport._host, args[1])

def test_delete_private_connection_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_private_connection(vmwareengine.DeletePrivateConnectionRequest(), name='name_value')

def test_delete_private_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vmwareengine.ListPrivateConnectionPeeringRoutesRequest, dict])
def test_list_private_connection_peering_routes_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/privateConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_private_connection_peering_routes(request)
    assert isinstance(response, pagers.ListPrivateConnectionPeeringRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_private_connection_peering_routes_rest_required_fields(request_type=vmwareengine.ListPrivateConnectionPeeringRoutesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VmwareEngineRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_private_connection_peering_routes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_private_connection_peering_routes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_private_connection_peering_routes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_private_connection_peering_routes_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_private_connection_peering_routes._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_private_connection_peering_routes_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VmwareEngineRestInterceptor())
    client = VmwareEngineClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VmwareEngineRestInterceptor, 'post_list_private_connection_peering_routes') as post, mock.patch.object(transports.VmwareEngineRestInterceptor, 'pre_list_private_connection_peering_routes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vmwareengine.ListPrivateConnectionPeeringRoutesRequest.pb(vmwareengine.ListPrivateConnectionPeeringRoutesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vmwareengine.ListPrivateConnectionPeeringRoutesResponse.to_json(vmwareengine.ListPrivateConnectionPeeringRoutesResponse())
        request = vmwareengine.ListPrivateConnectionPeeringRoutesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse()
        client.list_private_connection_peering_routes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_private_connection_peering_routes_rest_bad_request(transport: str='rest', request_type=vmwareengine.ListPrivateConnectionPeeringRoutesRequest):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/privateConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_private_connection_peering_routes(request)

def test_list_private_connection_peering_routes_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateConnections/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vmwareengine.ListPrivateConnectionPeeringRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_private_connection_peering_routes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/privateConnections/*}/peeringRoutes' % client.transport._host, args[1])

def test_list_private_connection_peering_routes_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_private_connection_peering_routes(vmwareengine.ListPrivateConnectionPeeringRoutesRequest(), parent='parent_value')

def test_list_private_connection_peering_routes_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()], next_page_token='abc'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[], next_page_token='def'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute()], next_page_token='ghi'), vmwareengine.ListPrivateConnectionPeeringRoutesResponse(peering_routes=[vmwareengine_resources.PeeringRoute(), vmwareengine_resources.PeeringRoute()]))
        response = response + response
        response = tuple((vmwareengine.ListPrivateConnectionPeeringRoutesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/privateConnections/sample3'}
        pager = client.list_private_connection_peering_routes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vmwareengine_resources.PeeringRoute) for i in results))
        pages = list(client.list_private_connection_peering_routes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VmwareEngineGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.VmwareEngineGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VmwareEngineClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.VmwareEngineGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VmwareEngineClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VmwareEngineClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.VmwareEngineGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VmwareEngineClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.VmwareEngineGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = VmwareEngineClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.VmwareEngineGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.VmwareEngineGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.VmwareEngineGrpcTransport, transports.VmwareEngineGrpcAsyncIOTransport, transports.VmwareEngineRestTransport])
def test_transport_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = VmwareEngineClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.VmwareEngineGrpcTransport)

def test_vmware_engine_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.VmwareEngineTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_vmware_engine_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.vmwareengine_v1.services.vmware_engine.transports.VmwareEngineTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.VmwareEngineTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_private_clouds', 'get_private_cloud', 'create_private_cloud', 'update_private_cloud', 'delete_private_cloud', 'undelete_private_cloud', 'list_clusters', 'get_cluster', 'create_cluster', 'update_cluster', 'delete_cluster', 'list_subnets', 'get_subnet', 'update_subnet', 'list_node_types', 'get_node_type', 'show_nsx_credentials', 'show_vcenter_credentials', 'reset_nsx_credentials', 'reset_vcenter_credentials', 'create_hcx_activation_key', 'list_hcx_activation_keys', 'get_hcx_activation_key', 'get_network_policy', 'list_network_policies', 'create_network_policy', 'update_network_policy', 'delete_network_policy', 'create_vmware_engine_network', 'update_vmware_engine_network', 'delete_vmware_engine_network', 'get_vmware_engine_network', 'list_vmware_engine_networks', 'create_private_connection', 'get_private_connection', 'list_private_connections', 'update_private_connection', 'delete_private_connection', 'list_private_connection_peering_routes', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'delete_operation', 'list_operations')
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

def test_vmware_engine_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.vmwareengine_v1.services.vmware_engine.transports.VmwareEngineTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VmwareEngineTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_vmware_engine_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.vmwareengine_v1.services.vmware_engine.transports.VmwareEngineTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VmwareEngineTransport()
        adc.assert_called_once()

def test_vmware_engine_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        VmwareEngineClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.VmwareEngineGrpcTransport, transports.VmwareEngineGrpcAsyncIOTransport])
def test_vmware_engine_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.VmwareEngineGrpcTransport, transports.VmwareEngineGrpcAsyncIOTransport, transports.VmwareEngineRestTransport])
def test_vmware_engine_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.VmwareEngineGrpcTransport, grpc_helpers), (transports.VmwareEngineGrpcAsyncIOTransport, grpc_helpers_async)])
def test_vmware_engine_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('vmwareengine.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='vmwareengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.VmwareEngineGrpcTransport, transports.VmwareEngineGrpcAsyncIOTransport])
def test_vmware_engine_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_vmware_engine_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.VmwareEngineRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_vmware_engine_rest_lro_client():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_vmware_engine_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vmwareengine.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('vmwareengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmwareengine.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_vmware_engine_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vmwareengine.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('vmwareengine.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vmwareengine.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_vmware_engine_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = VmwareEngineClient(credentials=creds1, transport=transport_name)
    client2 = VmwareEngineClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_private_clouds._session
    session2 = client2.transport.list_private_clouds._session
    assert session1 != session2
    session1 = client1.transport.get_private_cloud._session
    session2 = client2.transport.get_private_cloud._session
    assert session1 != session2
    session1 = client1.transport.create_private_cloud._session
    session2 = client2.transport.create_private_cloud._session
    assert session1 != session2
    session1 = client1.transport.update_private_cloud._session
    session2 = client2.transport.update_private_cloud._session
    assert session1 != session2
    session1 = client1.transport.delete_private_cloud._session
    session2 = client2.transport.delete_private_cloud._session
    assert session1 != session2
    session1 = client1.transport.undelete_private_cloud._session
    session2 = client2.transport.undelete_private_cloud._session
    assert session1 != session2
    session1 = client1.transport.list_clusters._session
    session2 = client2.transport.list_clusters._session
    assert session1 != session2
    session1 = client1.transport.get_cluster._session
    session2 = client2.transport.get_cluster._session
    assert session1 != session2
    session1 = client1.transport.create_cluster._session
    session2 = client2.transport.create_cluster._session
    assert session1 != session2
    session1 = client1.transport.update_cluster._session
    session2 = client2.transport.update_cluster._session
    assert session1 != session2
    session1 = client1.transport.delete_cluster._session
    session2 = client2.transport.delete_cluster._session
    assert session1 != session2
    session1 = client1.transport.list_subnets._session
    session2 = client2.transport.list_subnets._session
    assert session1 != session2
    session1 = client1.transport.get_subnet._session
    session2 = client2.transport.get_subnet._session
    assert session1 != session2
    session1 = client1.transport.update_subnet._session
    session2 = client2.transport.update_subnet._session
    assert session1 != session2
    session1 = client1.transport.list_node_types._session
    session2 = client2.transport.list_node_types._session
    assert session1 != session2
    session1 = client1.transport.get_node_type._session
    session2 = client2.transport.get_node_type._session
    assert session1 != session2
    session1 = client1.transport.show_nsx_credentials._session
    session2 = client2.transport.show_nsx_credentials._session
    assert session1 != session2
    session1 = client1.transport.show_vcenter_credentials._session
    session2 = client2.transport.show_vcenter_credentials._session
    assert session1 != session2
    session1 = client1.transport.reset_nsx_credentials._session
    session2 = client2.transport.reset_nsx_credentials._session
    assert session1 != session2
    session1 = client1.transport.reset_vcenter_credentials._session
    session2 = client2.transport.reset_vcenter_credentials._session
    assert session1 != session2
    session1 = client1.transport.create_hcx_activation_key._session
    session2 = client2.transport.create_hcx_activation_key._session
    assert session1 != session2
    session1 = client1.transport.list_hcx_activation_keys._session
    session2 = client2.transport.list_hcx_activation_keys._session
    assert session1 != session2
    session1 = client1.transport.get_hcx_activation_key._session
    session2 = client2.transport.get_hcx_activation_key._session
    assert session1 != session2
    session1 = client1.transport.get_network_policy._session
    session2 = client2.transport.get_network_policy._session
    assert session1 != session2
    session1 = client1.transport.list_network_policies._session
    session2 = client2.transport.list_network_policies._session
    assert session1 != session2
    session1 = client1.transport.create_network_policy._session
    session2 = client2.transport.create_network_policy._session
    assert session1 != session2
    session1 = client1.transport.update_network_policy._session
    session2 = client2.transport.update_network_policy._session
    assert session1 != session2
    session1 = client1.transport.delete_network_policy._session
    session2 = client2.transport.delete_network_policy._session
    assert session1 != session2
    session1 = client1.transport.create_vmware_engine_network._session
    session2 = client2.transport.create_vmware_engine_network._session
    assert session1 != session2
    session1 = client1.transport.update_vmware_engine_network._session
    session2 = client2.transport.update_vmware_engine_network._session
    assert session1 != session2
    session1 = client1.transport.delete_vmware_engine_network._session
    session2 = client2.transport.delete_vmware_engine_network._session
    assert session1 != session2
    session1 = client1.transport.get_vmware_engine_network._session
    session2 = client2.transport.get_vmware_engine_network._session
    assert session1 != session2
    session1 = client1.transport.list_vmware_engine_networks._session
    session2 = client2.transport.list_vmware_engine_networks._session
    assert session1 != session2
    session1 = client1.transport.create_private_connection._session
    session2 = client2.transport.create_private_connection._session
    assert session1 != session2
    session1 = client1.transport.get_private_connection._session
    session2 = client2.transport.get_private_connection._session
    assert session1 != session2
    session1 = client1.transport.list_private_connections._session
    session2 = client2.transport.list_private_connections._session
    assert session1 != session2
    session1 = client1.transport.update_private_connection._session
    session2 = client2.transport.update_private_connection._session
    assert session1 != session2
    session1 = client1.transport.delete_private_connection._session
    session2 = client2.transport.delete_private_connection._session
    assert session1 != session2
    session1 = client1.transport.list_private_connection_peering_routes._session
    session2 = client2.transport.list_private_connection_peering_routes._session
    assert session1 != session2

def test_vmware_engine_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VmwareEngineGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_vmware_engine_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VmwareEngineGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.VmwareEngineGrpcTransport, transports.VmwareEngineGrpcAsyncIOTransport])
def test_vmware_engine_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.VmwareEngineGrpcTransport, transports.VmwareEngineGrpcAsyncIOTransport])
def test_vmware_engine_transport_channel_mtls_with_adc(transport_class):
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

def test_vmware_engine_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_vmware_engine_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_cluster_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    private_cloud = 'whelk'
    cluster = 'octopus'
    expected = 'projects/{project}/locations/{location}/privateClouds/{private_cloud}/clusters/{cluster}'.format(project=project, location=location, private_cloud=private_cloud, cluster=cluster)
    actual = VmwareEngineClient.cluster_path(project, location, private_cloud, cluster)
    assert expected == actual

def test_parse_cluster_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'private_cloud': 'cuttlefish', 'cluster': 'mussel'}
    path = VmwareEngineClient.cluster_path(**expected)
    actual = VmwareEngineClient.parse_cluster_path(path)
    assert expected == actual

def test_hcx_activation_key_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    private_cloud = 'scallop'
    hcx_activation_key = 'abalone'
    expected = 'projects/{project}/locations/{location}/privateClouds/{private_cloud}/hcxActivationKeys/{hcx_activation_key}'.format(project=project, location=location, private_cloud=private_cloud, hcx_activation_key=hcx_activation_key)
    actual = VmwareEngineClient.hcx_activation_key_path(project, location, private_cloud, hcx_activation_key)
    assert expected == actual

def test_parse_hcx_activation_key_path():
    if False:
        print('Hello World!')
    expected = {'project': 'squid', 'location': 'clam', 'private_cloud': 'whelk', 'hcx_activation_key': 'octopus'}
    path = VmwareEngineClient.hcx_activation_key_path(**expected)
    actual = VmwareEngineClient.parse_hcx_activation_key_path(path)
    assert expected == actual

def test_network_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    network = 'nudibranch'
    expected = 'projects/{project}/global/networks/{network}'.format(project=project, network=network)
    actual = VmwareEngineClient.network_path(project, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        return 10
    expected = {'project': 'cuttlefish', 'network': 'mussel'}
    path = VmwareEngineClient.network_path(**expected)
    actual = VmwareEngineClient.parse_network_path(path)
    assert expected == actual

def test_network_policy_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    network_policy = 'scallop'
    expected = 'projects/{project}/locations/{location}/networkPolicies/{network_policy}'.format(project=project, location=location, network_policy=network_policy)
    actual = VmwareEngineClient.network_policy_path(project, location, network_policy)
    assert expected == actual

def test_parse_network_policy_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'network_policy': 'clam'}
    path = VmwareEngineClient.network_policy_path(**expected)
    actual = VmwareEngineClient.parse_network_policy_path(path)
    assert expected == actual

def test_node_type_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    node_type = 'oyster'
    expected = 'projects/{project}/locations/{location}/nodeTypes/{node_type}'.format(project=project, location=location, node_type=node_type)
    actual = VmwareEngineClient.node_type_path(project, location, node_type)
    assert expected == actual

def test_parse_node_type_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'node_type': 'mussel'}
    path = VmwareEngineClient.node_type_path(**expected)
    actual = VmwareEngineClient.parse_node_type_path(path)
    assert expected == actual

def test_private_cloud_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    private_cloud = 'scallop'
    expected = 'projects/{project}/locations/{location}/privateClouds/{private_cloud}'.format(project=project, location=location, private_cloud=private_cloud)
    actual = VmwareEngineClient.private_cloud_path(project, location, private_cloud)
    assert expected == actual

def test_parse_private_cloud_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'private_cloud': 'clam'}
    path = VmwareEngineClient.private_cloud_path(**expected)
    actual = VmwareEngineClient.parse_private_cloud_path(path)
    assert expected == actual

def test_private_connection_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    private_connection = 'oyster'
    expected = 'projects/{project}/locations/{location}/privateConnections/{private_connection}'.format(project=project, location=location, private_connection=private_connection)
    actual = VmwareEngineClient.private_connection_path(project, location, private_connection)
    assert expected == actual

def test_parse_private_connection_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'private_connection': 'mussel'}
    path = VmwareEngineClient.private_connection_path(**expected)
    actual = VmwareEngineClient.parse_private_connection_path(path)
    assert expected == actual

def test_subnet_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    private_cloud = 'scallop'
    subnet = 'abalone'
    expected = 'projects/{project}/locations/{location}/privateClouds/{private_cloud}/subnets/{subnet}'.format(project=project, location=location, private_cloud=private_cloud, subnet=subnet)
    actual = VmwareEngineClient.subnet_path(project, location, private_cloud, subnet)
    assert expected == actual

def test_parse_subnet_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam', 'private_cloud': 'whelk', 'subnet': 'octopus'}
    path = VmwareEngineClient.subnet_path(**expected)
    actual = VmwareEngineClient.parse_subnet_path(path)
    assert expected == actual

def test_vmware_engine_network_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    vmware_engine_network = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/vmwareEngineNetworks/{vmware_engine_network}'.format(project=project, location=location, vmware_engine_network=vmware_engine_network)
    actual = VmwareEngineClient.vmware_engine_network_path(project, location, vmware_engine_network)
    assert expected == actual

def test_parse_vmware_engine_network_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel', 'location': 'winkle', 'vmware_engine_network': 'nautilus'}
    path = VmwareEngineClient.vmware_engine_network_path(**expected)
    actual = VmwareEngineClient.parse_vmware_engine_network_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = VmwareEngineClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'abalone'}
    path = VmwareEngineClient.common_billing_account_path(**expected)
    actual = VmwareEngineClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = VmwareEngineClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'clam'}
    path = VmwareEngineClient.common_folder_path(**expected)
    actual = VmwareEngineClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = VmwareEngineClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'octopus'}
    path = VmwareEngineClient.common_organization_path(**expected)
    actual = VmwareEngineClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = VmwareEngineClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch'}
    path = VmwareEngineClient.common_project_path(**expected)
    actual = VmwareEngineClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = VmwareEngineClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = VmwareEngineClient.common_location_path(**expected)
    actual = VmwareEngineClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.VmwareEngineTransport, '_prep_wrapped_messages') as prep:
        client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.VmwareEngineTransport, '_prep_wrapped_messages') as prep:
        transport_class = VmwareEngineClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_location(request)

@pytest.mark.parametrize('request_type', [locations_pb2.GetLocationRequest, dict])
def test_get_location_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = locations_pb2.Location()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_location(request)
    assert isinstance(response, locations_pb2.Location)

def test_list_locations_rest_bad_request(transport: str='rest', request_type=locations_pb2.ListLocationsRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_locations(request)

@pytest.mark.parametrize('request_type', [locations_pb2.ListLocationsRequest, dict])
def test_list_locations_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = locations_pb2.ListLocationsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_locations(request)
    assert isinstance(response, locations_pb2.ListLocationsResponse)

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/privateClouds/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/privateClouds/sample3'}
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
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/privateClouds/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/privateClouds/sample3'}
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
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/privateClouds/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/privateClouds/sample3'}
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

def test_delete_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.DeleteOperationRequest):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
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
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.ListLocationsRequest()
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.ListLocationsResponse)

@pytest.mark.asyncio
async def test_list_locations_async(transport: str='grpc'):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.ListLocationsRequest()
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.ListLocationsResponse)

def test_list_locations_field_headers():
    if False:
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.ListLocationsRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_list_locations_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.ListLocationsRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        await client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_list_locations_from_dict():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.GetLocationRequest()
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.Location)

@pytest.mark.asyncio
async def test_get_location_async(transport: str='grpc_asyncio'):
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.GetLocationRequest()
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.Location)

def test_get_location_field_headers():
    if False:
        return 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.GetLocationRequest()
    request.name = 'locations/abc'
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = locations_pb2.Location()
        client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations/abc') in kw['metadata']

@pytest.mark.asyncio
async def test_get_location_field_headers_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.GetLocationRequest()
    request.name = 'locations/abc'
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        await client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations/abc') in kw['metadata']

def test_get_location_from_dict():
    if False:
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = VmwareEngineAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = VmwareEngineClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(VmwareEngineClient, transports.VmwareEngineGrpcTransport), (VmwareEngineAsyncClient, transports.VmwareEngineGrpcAsyncIOTransport)])
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