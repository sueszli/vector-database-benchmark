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
from google.cloud.bare_metal_solution_v2.services.bare_metal_solution import BareMetalSolutionAsyncClient, BareMetalSolutionClient, pagers, transports
from google.cloud.bare_metal_solution_v2.types import nfs_share as gcb_nfs_share
from google.cloud.bare_metal_solution_v2.types import volume_snapshot as gcb_volume_snapshot
from google.cloud.bare_metal_solution_v2.types import baremetalsolution, common
from google.cloud.bare_metal_solution_v2.types import instance
from google.cloud.bare_metal_solution_v2.types import instance as gcb_instance
from google.cloud.bare_metal_solution_v2.types import lun
from google.cloud.bare_metal_solution_v2.types import network
from google.cloud.bare_metal_solution_v2.types import network as gcb_network
from google.cloud.bare_metal_solution_v2.types import nfs_share
from google.cloud.bare_metal_solution_v2.types import osimage, provisioning
from google.cloud.bare_metal_solution_v2.types import ssh_key
from google.cloud.bare_metal_solution_v2.types import ssh_key as gcb_ssh_key
from google.cloud.bare_metal_solution_v2.types import volume
from google.cloud.bare_metal_solution_v2.types import volume as gcb_volume
from google.cloud.bare_metal_solution_v2.types import volume_snapshot

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
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert BareMetalSolutionClient._get_default_mtls_endpoint(None) is None
    assert BareMetalSolutionClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert BareMetalSolutionClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert BareMetalSolutionClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert BareMetalSolutionClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert BareMetalSolutionClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(BareMetalSolutionClient, 'grpc'), (BareMetalSolutionAsyncClient, 'grpc_asyncio'), (BareMetalSolutionClient, 'rest')])
def test_bare_metal_solution_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('baremetalsolution.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://baremetalsolution.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.BareMetalSolutionGrpcTransport, 'grpc'), (transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.BareMetalSolutionRestTransport, 'rest')])
def test_bare_metal_solution_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(BareMetalSolutionClient, 'grpc'), (BareMetalSolutionAsyncClient, 'grpc_asyncio'), (BareMetalSolutionClient, 'rest')])
def test_bare_metal_solution_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('baremetalsolution.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://baremetalsolution.googleapis.com')

def test_bare_metal_solution_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = BareMetalSolutionClient.get_transport_class()
    available_transports = [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionRestTransport]
    assert transport in available_transports
    transport = BareMetalSolutionClient.get_transport_class('grpc')
    assert transport == transports.BareMetalSolutionGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport, 'grpc'), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio'), (BareMetalSolutionClient, transports.BareMetalSolutionRestTransport, 'rest')])
@mock.patch.object(BareMetalSolutionClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BareMetalSolutionClient))
@mock.patch.object(BareMetalSolutionAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BareMetalSolutionAsyncClient))
def test_bare_metal_solution_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(BareMetalSolutionClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(BareMetalSolutionClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport, 'grpc', 'true'), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport, 'grpc', 'false'), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (BareMetalSolutionClient, transports.BareMetalSolutionRestTransport, 'rest', 'true'), (BareMetalSolutionClient, transports.BareMetalSolutionRestTransport, 'rest', 'false')])
@mock.patch.object(BareMetalSolutionClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BareMetalSolutionClient))
@mock.patch.object(BareMetalSolutionAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BareMetalSolutionAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_bare_metal_solution_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [BareMetalSolutionClient, BareMetalSolutionAsyncClient])
@mock.patch.object(BareMetalSolutionClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BareMetalSolutionClient))
@mock.patch.object(BareMetalSolutionAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BareMetalSolutionAsyncClient))
def test_bare_metal_solution_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport, 'grpc'), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio'), (BareMetalSolutionClient, transports.BareMetalSolutionRestTransport, 'rest')])
def test_bare_metal_solution_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport, 'grpc', grpc_helpers), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (BareMetalSolutionClient, transports.BareMetalSolutionRestTransport, 'rest', None)])
def test_bare_metal_solution_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_bare_metal_solution_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.bare_metal_solution_v2.services.bare_metal_solution.transports.BareMetalSolutionGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = BareMetalSolutionClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport, 'grpc', grpc_helpers), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_bare_metal_solution_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('baremetalsolution.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='baremetalsolution.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [instance.ListInstancesRequest, dict])
def test_list_instances(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = instance.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        client.list_instances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.ListInstancesRequest()

@pytest.mark.asyncio
async def test_list_instances_async(transport: str='grpc_asyncio', request_type=instance.ListInstancesRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_instances_async_from_dict():
    await test_list_instances_async(request_type=dict)

def test_list_instances_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = instance.ListInstancesResponse()
        client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_instances_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.ListInstancesResponse())
        await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_instances_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = instance.ListInstancesResponse()
        client.list_instances(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_instances_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_instances(instance.ListInstancesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_instances_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = instance.ListInstancesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.ListInstancesResponse())
        response = await client.list_instances(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_instances_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_instances(instance.ListInstancesRequest(), parent='parent_value')

def test_list_instances_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance(), instance.Instance()], next_page_token='abc'), instance.ListInstancesResponse(instances=[], next_page_token='def'), instance.ListInstancesResponse(instances=[instance.Instance()], next_page_token='ghi'), instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_instances(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, instance.Instance) for i in results))

def test_list_instances_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance(), instance.Instance()], next_page_token='abc'), instance.ListInstancesResponse(instances=[], next_page_token='def'), instance.ListInstancesResponse(instances=[instance.Instance()], next_page_token='ghi'), instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance()]), RuntimeError)
        pages = list(client.list_instances(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_instances_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance(), instance.Instance()], next_page_token='abc'), instance.ListInstancesResponse(instances=[], next_page_token='def'), instance.ListInstancesResponse(instances=[instance.Instance()], next_page_token='ghi'), instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance()]), RuntimeError)
        async_pager = await client.list_instances(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, instance.Instance) for i in responses))

@pytest.mark.asyncio
async def test_list_instances_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance(), instance.Instance()], next_page_token='abc'), instance.ListInstancesResponse(instances=[], next_page_token='def'), instance.ListInstancesResponse(instances=[instance.Instance()], next_page_token='ghi'), instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_instances(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [instance.GetInstanceRequest, dict])
def test_get_instance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = instance.Instance(name='name_value', id='id_value', machine_type='machine_type_value', state=instance.Instance.State.PROVISIONING, hyperthreading_enabled=True, interactive_serial_console_enabled=True, os_image='os_image_value', pod='pod_value', network_template='network_template_value', login_info='login_info_value', workload_profile=common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC, firmware_version='firmware_version_value')
        response = client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.GetInstanceRequest()
    assert isinstance(response, instance.Instance)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.machine_type == 'machine_type_value'
    assert response.state == instance.Instance.State.PROVISIONING
    assert response.hyperthreading_enabled is True
    assert response.interactive_serial_console_enabled is True
    assert response.os_image == 'os_image_value'
    assert response.pod == 'pod_value'
    assert response.network_template == 'network_template_value'
    assert response.login_info == 'login_info_value'
    assert response.workload_profile == common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC
    assert response.firmware_version == 'firmware_version_value'

def test_get_instance_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        client.get_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.GetInstanceRequest()

@pytest.mark.asyncio
async def test_get_instance_async(transport: str='grpc_asyncio', request_type=instance.GetInstanceRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.Instance(name='name_value', id='id_value', machine_type='machine_type_value', state=instance.Instance.State.PROVISIONING, hyperthreading_enabled=True, interactive_serial_console_enabled=True, os_image='os_image_value', pod='pod_value', network_template='network_template_value', login_info='login_info_value', workload_profile=common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC, firmware_version='firmware_version_value'))
        response = await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.GetInstanceRequest()
    assert isinstance(response, instance.Instance)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.machine_type == 'machine_type_value'
    assert response.state == instance.Instance.State.PROVISIONING
    assert response.hyperthreading_enabled is True
    assert response.interactive_serial_console_enabled is True
    assert response.os_image == 'os_image_value'
    assert response.pod == 'pod_value'
    assert response.network_template == 'network_template_value'
    assert response.login_info == 'login_info_value'
    assert response.workload_profile == common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC
    assert response.firmware_version == 'firmware_version_value'

@pytest.mark.asyncio
async def test_get_instance_async_from_dict():
    await test_get_instance_async(request_type=dict)

def test_get_instance_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = instance.Instance()
        client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.Instance())
        await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_instance_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = instance.Instance()
        client.get_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance(instance.GetInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = instance.Instance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.Instance())
        response = await client.get_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance(instance.GetInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcb_instance.UpdateInstanceRequest, dict])
def test_update_instance(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_instance.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

def test_update_instance_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        client.update_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_instance.UpdateInstanceRequest()

@pytest.mark.asyncio
async def test_update_instance_async(transport: str='grpc_asyncio', request_type=gcb_instance.UpdateInstanceRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_instance.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_instance_async_from_dict():
    await test_update_instance_async(request_type=dict)

def test_update_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_instance.UpdateInstanceRequest()
    request.instance.name = 'name_value'
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_instance_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_instance.UpdateInstanceRequest()
    request.instance.name = 'name_value'
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance.name=name_value') in kw['metadata']

def test_update_instance_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(instance=gcb_instance.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = gcb_instance.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_instance(gcb_instance.UpdateInstanceRequest(), instance=gcb_instance.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_instance_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(instance=gcb_instance.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = gcb_instance.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_instance_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_instance(gcb_instance.UpdateInstanceRequest(), instance=gcb_instance.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [instance.RenameInstanceRequest, dict])
def test_rename_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        call.return_value = instance.Instance(name='name_value', id='id_value', machine_type='machine_type_value', state=instance.Instance.State.PROVISIONING, hyperthreading_enabled=True, interactive_serial_console_enabled=True, os_image='os_image_value', pod='pod_value', network_template='network_template_value', login_info='login_info_value', workload_profile=common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC, firmware_version='firmware_version_value')
        response = client.rename_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.RenameInstanceRequest()
    assert isinstance(response, instance.Instance)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.machine_type == 'machine_type_value'
    assert response.state == instance.Instance.State.PROVISIONING
    assert response.hyperthreading_enabled is True
    assert response.interactive_serial_console_enabled is True
    assert response.os_image == 'os_image_value'
    assert response.pod == 'pod_value'
    assert response.network_template == 'network_template_value'
    assert response.login_info == 'login_info_value'
    assert response.workload_profile == common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC
    assert response.firmware_version == 'firmware_version_value'

def test_rename_instance_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        client.rename_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.RenameInstanceRequest()

@pytest.mark.asyncio
async def test_rename_instance_async(transport: str='grpc_asyncio', request_type=instance.RenameInstanceRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.Instance(name='name_value', id='id_value', machine_type='machine_type_value', state=instance.Instance.State.PROVISIONING, hyperthreading_enabled=True, interactive_serial_console_enabled=True, os_image='os_image_value', pod='pod_value', network_template='network_template_value', login_info='login_info_value', workload_profile=common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC, firmware_version='firmware_version_value'))
        response = await client.rename_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.RenameInstanceRequest()
    assert isinstance(response, instance.Instance)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.machine_type == 'machine_type_value'
    assert response.state == instance.Instance.State.PROVISIONING
    assert response.hyperthreading_enabled is True
    assert response.interactive_serial_console_enabled is True
    assert response.os_image == 'os_image_value'
    assert response.pod == 'pod_value'
    assert response.network_template == 'network_template_value'
    assert response.login_info == 'login_info_value'
    assert response.workload_profile == common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC
    assert response.firmware_version == 'firmware_version_value'

@pytest.mark.asyncio
async def test_rename_instance_async_from_dict():
    await test_rename_instance_async(request_type=dict)

def test_rename_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.RenameInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        call.return_value = instance.Instance()
        client.rename_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_rename_instance_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.RenameInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.Instance())
        await client.rename_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_rename_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        call.return_value = instance.Instance()
        client.rename_instance(name='name_value', new_instance_id='new_instance_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_instance_id
        mock_val = 'new_instance_id_value'
        assert arg == mock_val

def test_rename_instance_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.rename_instance(instance.RenameInstanceRequest(), name='name_value', new_instance_id='new_instance_id_value')

@pytest.mark.asyncio
async def test_rename_instance_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_instance), '__call__') as call:
        call.return_value = instance.Instance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance.Instance())
        response = await client.rename_instance(name='name_value', new_instance_id='new_instance_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_instance_id
        mock_val = 'new_instance_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_rename_instance_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.rename_instance(instance.RenameInstanceRequest(), name='name_value', new_instance_id='new_instance_id_value')

@pytest.mark.parametrize('request_type', [instance.ResetInstanceRequest, dict])
def test_reset_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reset_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.ResetInstanceRequest()
    assert isinstance(response, future.Future)

def test_reset_instance_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        client.reset_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.ResetInstanceRequest()

@pytest.mark.asyncio
async def test_reset_instance_async(transport: str='grpc_asyncio', request_type=instance.ResetInstanceRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.ResetInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reset_instance_async_from_dict():
    await test_reset_instance_async(request_type=dict)

def test_reset_instance_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.ResetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reset_instance_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.ResetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reset_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_reset_instance_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_reset_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reset_instance(instance.ResetInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_reset_instance_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reset_instance_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reset_instance(instance.ResetInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [instance.StartInstanceRequest, dict])
def test_start_instance(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.StartInstanceRequest()
    assert isinstance(response, future.Future)

def test_start_instance_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        client.start_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.StartInstanceRequest()

@pytest.mark.asyncio
async def test_start_instance_async(transport: str='grpc_asyncio', request_type=instance.StartInstanceRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.StartInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_instance_async_from_dict():
    await test_start_instance_async(request_type=dict)

def test_start_instance_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.StartInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_instance_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.StartInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_start_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_start_instance_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.start_instance(instance.StartInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_start_instance_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_start_instance_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.start_instance(instance.StartInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [instance.StopInstanceRequest, dict])
def test_stop_instance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.stop_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.StopInstanceRequest()
    assert isinstance(response, future.Future)

def test_stop_instance_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        client.stop_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.StopInstanceRequest()

@pytest.mark.asyncio
async def test_stop_instance_async(transport: str='grpc_asyncio', request_type=instance.StopInstanceRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.StopInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_stop_instance_async_from_dict():
    await test_stop_instance_async(request_type=dict)

def test_stop_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.StopInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_stop_instance_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.StopInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.stop_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_stop_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_stop_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.stop_instance(instance.StopInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_stop_instance_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_stop_instance_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.stop_instance(instance.StopInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [instance.EnableInteractiveSerialConsoleRequest, dict])
def test_enable_interactive_serial_console(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.enable_interactive_serial_console(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.EnableInteractiveSerialConsoleRequest()
    assert isinstance(response, future.Future)

def test_enable_interactive_serial_console_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        client.enable_interactive_serial_console()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.EnableInteractiveSerialConsoleRequest()

@pytest.mark.asyncio
async def test_enable_interactive_serial_console_async(transport: str='grpc_asyncio', request_type=instance.EnableInteractiveSerialConsoleRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.enable_interactive_serial_console(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.EnableInteractiveSerialConsoleRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_enable_interactive_serial_console_async_from_dict():
    await test_enable_interactive_serial_console_async(request_type=dict)

def test_enable_interactive_serial_console_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.EnableInteractiveSerialConsoleRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.enable_interactive_serial_console(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_enable_interactive_serial_console_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.EnableInteractiveSerialConsoleRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.enable_interactive_serial_console(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_enable_interactive_serial_console_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.enable_interactive_serial_console(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_enable_interactive_serial_console_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.enable_interactive_serial_console(instance.EnableInteractiveSerialConsoleRequest(), name='name_value')

@pytest.mark.asyncio
async def test_enable_interactive_serial_console_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.enable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.enable_interactive_serial_console(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_enable_interactive_serial_console_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.enable_interactive_serial_console(instance.EnableInteractiveSerialConsoleRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [instance.DisableInteractiveSerialConsoleRequest, dict])
def test_disable_interactive_serial_console(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.disable_interactive_serial_console(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.DisableInteractiveSerialConsoleRequest()
    assert isinstance(response, future.Future)

def test_disable_interactive_serial_console_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        client.disable_interactive_serial_console()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.DisableInteractiveSerialConsoleRequest()

@pytest.mark.asyncio
async def test_disable_interactive_serial_console_async(transport: str='grpc_asyncio', request_type=instance.DisableInteractiveSerialConsoleRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.disable_interactive_serial_console(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance.DisableInteractiveSerialConsoleRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_disable_interactive_serial_console_async_from_dict():
    await test_disable_interactive_serial_console_async(request_type=dict)

def test_disable_interactive_serial_console_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.DisableInteractiveSerialConsoleRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.disable_interactive_serial_console(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_disable_interactive_serial_console_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance.DisableInteractiveSerialConsoleRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.disable_interactive_serial_console(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_disable_interactive_serial_console_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.disable_interactive_serial_console(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_disable_interactive_serial_console_flattened_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.disable_interactive_serial_console(instance.DisableInteractiveSerialConsoleRequest(), name='name_value')

@pytest.mark.asyncio
async def test_disable_interactive_serial_console_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.disable_interactive_serial_console), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.disable_interactive_serial_console(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_disable_interactive_serial_console_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.disable_interactive_serial_console(instance.DisableInteractiveSerialConsoleRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcb_instance.DetachLunRequest, dict])
def test_detach_lun(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.detach_lun(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_instance.DetachLunRequest()
    assert isinstance(response, future.Future)

def test_detach_lun_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        client.detach_lun()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_instance.DetachLunRequest()

@pytest.mark.asyncio
async def test_detach_lun_async(transport: str='grpc_asyncio', request_type=gcb_instance.DetachLunRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.detach_lun(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_instance.DetachLunRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_detach_lun_async_from_dict():
    await test_detach_lun_async(request_type=dict)

def test_detach_lun_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_instance.DetachLunRequest()
    request.instance = 'instance_value'
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.detach_lun(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance=instance_value') in kw['metadata']

@pytest.mark.asyncio
async def test_detach_lun_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_instance.DetachLunRequest()
    request.instance = 'instance_value'
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.detach_lun(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance=instance_value') in kw['metadata']

def test_detach_lun_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.detach_lun(instance='instance_value', lun='lun_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = 'instance_value'
        assert arg == mock_val
        arg = args[0].lun
        mock_val = 'lun_value'
        assert arg == mock_val

def test_detach_lun_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.detach_lun(gcb_instance.DetachLunRequest(), instance='instance_value', lun='lun_value')

@pytest.mark.asyncio
async def test_detach_lun_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.detach_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.detach_lun(instance='instance_value', lun='lun_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = 'instance_value'
        assert arg == mock_val
        arg = args[0].lun
        mock_val = 'lun_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_detach_lun_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.detach_lun(gcb_instance.DetachLunRequest(), instance='instance_value', lun='lun_value')

@pytest.mark.parametrize('request_type', [ssh_key.ListSSHKeysRequest, dict])
def test_list_ssh_keys(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.return_value = ssh_key.ListSSHKeysResponse(next_page_token='next_page_token_value')
        response = client.list_ssh_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ssh_key.ListSSHKeysRequest()
    assert isinstance(response, pagers.ListSSHKeysPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_ssh_keys_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        client.list_ssh_keys()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ssh_key.ListSSHKeysRequest()

@pytest.mark.asyncio
async def test_list_ssh_keys_async(transport: str='grpc_asyncio', request_type=ssh_key.ListSSHKeysRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ssh_key.ListSSHKeysResponse(next_page_token='next_page_token_value'))
        response = await client.list_ssh_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ssh_key.ListSSHKeysRequest()
    assert isinstance(response, pagers.ListSSHKeysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_ssh_keys_async_from_dict():
    await test_list_ssh_keys_async(request_type=dict)

def test_list_ssh_keys_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = ssh_key.ListSSHKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.return_value = ssh_key.ListSSHKeysResponse()
        client.list_ssh_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_ssh_keys_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ssh_key.ListSSHKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ssh_key.ListSSHKeysResponse())
        await client.list_ssh_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_ssh_keys_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.return_value = ssh_key.ListSSHKeysResponse()
        client.list_ssh_keys(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_ssh_keys_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_ssh_keys(ssh_key.ListSSHKeysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_ssh_keys_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.return_value = ssh_key.ListSSHKeysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ssh_key.ListSSHKeysResponse())
        response = await client.list_ssh_keys(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_ssh_keys_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_ssh_keys(ssh_key.ListSSHKeysRequest(), parent='parent_value')

def test_list_ssh_keys_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.side_effect = (ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey(), ssh_key.SSHKey()], next_page_token='abc'), ssh_key.ListSSHKeysResponse(ssh_keys=[], next_page_token='def'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey()], next_page_token='ghi'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_ssh_keys(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, ssh_key.SSHKey) for i in results))

def test_list_ssh_keys_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__') as call:
        call.side_effect = (ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey(), ssh_key.SSHKey()], next_page_token='abc'), ssh_key.ListSSHKeysResponse(ssh_keys=[], next_page_token='def'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey()], next_page_token='ghi'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey()]), RuntimeError)
        pages = list(client.list_ssh_keys(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_ssh_keys_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey(), ssh_key.SSHKey()], next_page_token='abc'), ssh_key.ListSSHKeysResponse(ssh_keys=[], next_page_token='def'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey()], next_page_token='ghi'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey()]), RuntimeError)
        async_pager = await client.list_ssh_keys(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, ssh_key.SSHKey) for i in responses))

@pytest.mark.asyncio
async def test_list_ssh_keys_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_ssh_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey(), ssh_key.SSHKey()], next_page_token='abc'), ssh_key.ListSSHKeysResponse(ssh_keys=[], next_page_token='def'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey()], next_page_token='ghi'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_ssh_keys(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcb_ssh_key.CreateSSHKeyRequest, dict])
def test_create_ssh_key(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        call.return_value = gcb_ssh_key.SSHKey(name='name_value', public_key='public_key_value')
        response = client.create_ssh_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_ssh_key.CreateSSHKeyRequest()
    assert isinstance(response, gcb_ssh_key.SSHKey)
    assert response.name == 'name_value'
    assert response.public_key == 'public_key_value'

def test_create_ssh_key_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        client.create_ssh_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_ssh_key.CreateSSHKeyRequest()

@pytest.mark.asyncio
async def test_create_ssh_key_async(transport: str='grpc_asyncio', request_type=gcb_ssh_key.CreateSSHKeyRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcb_ssh_key.SSHKey(name='name_value', public_key='public_key_value'))
        response = await client.create_ssh_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_ssh_key.CreateSSHKeyRequest()
    assert isinstance(response, gcb_ssh_key.SSHKey)
    assert response.name == 'name_value'
    assert response.public_key == 'public_key_value'

@pytest.mark.asyncio
async def test_create_ssh_key_async_from_dict():
    await test_create_ssh_key_async(request_type=dict)

def test_create_ssh_key_field_headers():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_ssh_key.CreateSSHKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        call.return_value = gcb_ssh_key.SSHKey()
        client.create_ssh_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_ssh_key_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_ssh_key.CreateSSHKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcb_ssh_key.SSHKey())
        await client.create_ssh_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_ssh_key_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        call.return_value = gcb_ssh_key.SSHKey()
        client.create_ssh_key(parent='parent_value', ssh_key=gcb_ssh_key.SSHKey(name='name_value'), ssh_key_id='ssh_key_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ssh_key
        mock_val = gcb_ssh_key.SSHKey(name='name_value')
        assert arg == mock_val
        arg = args[0].ssh_key_id
        mock_val = 'ssh_key_id_value'
        assert arg == mock_val

def test_create_ssh_key_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_ssh_key(gcb_ssh_key.CreateSSHKeyRequest(), parent='parent_value', ssh_key=gcb_ssh_key.SSHKey(name='name_value'), ssh_key_id='ssh_key_id_value')

@pytest.mark.asyncio
async def test_create_ssh_key_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_ssh_key), '__call__') as call:
        call.return_value = gcb_ssh_key.SSHKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcb_ssh_key.SSHKey())
        response = await client.create_ssh_key(parent='parent_value', ssh_key=gcb_ssh_key.SSHKey(name='name_value'), ssh_key_id='ssh_key_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ssh_key
        mock_val = gcb_ssh_key.SSHKey(name='name_value')
        assert arg == mock_val
        arg = args[0].ssh_key_id
        mock_val = 'ssh_key_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_ssh_key_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_ssh_key(gcb_ssh_key.CreateSSHKeyRequest(), parent='parent_value', ssh_key=gcb_ssh_key.SSHKey(name='name_value'), ssh_key_id='ssh_key_id_value')

@pytest.mark.parametrize('request_type', [ssh_key.DeleteSSHKeyRequest, dict])
def test_delete_ssh_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        call.return_value = None
        response = client.delete_ssh_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ssh_key.DeleteSSHKeyRequest()
    assert response is None

def test_delete_ssh_key_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        client.delete_ssh_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ssh_key.DeleteSSHKeyRequest()

@pytest.mark.asyncio
async def test_delete_ssh_key_async(transport: str='grpc_asyncio', request_type=ssh_key.DeleteSSHKeyRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_ssh_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ssh_key.DeleteSSHKeyRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_ssh_key_async_from_dict():
    await test_delete_ssh_key_async(request_type=dict)

def test_delete_ssh_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = ssh_key.DeleteSSHKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        call.return_value = None
        client.delete_ssh_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_ssh_key_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ssh_key.DeleteSSHKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_ssh_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_ssh_key_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        call.return_value = None
        client.delete_ssh_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_ssh_key_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_ssh_key(ssh_key.DeleteSSHKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_ssh_key_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_ssh_key), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_ssh_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_ssh_key_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_ssh_key(ssh_key.DeleteSSHKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [volume.ListVolumesRequest, dict])
def test_list_volumes(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_volumes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.ListVolumesRequest()
    assert isinstance(response, pagers.ListVolumesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_volumes_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        client.list_volumes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.ListVolumesRequest()

@pytest.mark.asyncio
async def test_list_volumes_async(transport: str='grpc_asyncio', request_type=volume.ListVolumesRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.ListVolumesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_volumes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.ListVolumesRequest()
    assert isinstance(response, pagers.ListVolumesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_volumes_async_from_dict():
    await test_list_volumes_async(request_type=dict)

def test_list_volumes_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.ListVolumesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse()
        client.list_volumes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_volumes_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.ListVolumesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.ListVolumesResponse())
        await client.list_volumes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_volumes_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse()
        client.list_volumes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_volumes_flattened_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_volumes(volume.ListVolumesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_volumes_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.ListVolumesResponse())
        response = await client.list_volumes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_volumes_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_volumes(volume.ListVolumesRequest(), parent='parent_value')

def test_list_volumes_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_volumes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, volume.Volume) for i in results))

def test_list_volumes_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        pages = list(client.list_volumes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_volumes_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_volumes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        async_pager = await client.list_volumes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, volume.Volume) for i in responses))

@pytest.mark.asyncio
async def test_list_volumes_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_volumes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_volumes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [volume.GetVolumeRequest, dict])
def test_get_volume(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume(name='name_value', id='id_value', storage_type=volume.Volume.StorageType.SSD, state=volume.Volume.State.CREATING, requested_size_gib=1917, originally_requested_size_gib=3094, current_size_gib=1710, emergency_size_gib=1898, max_size_gib=1265, auto_grown_size_gib=2032, remaining_space_gib=1974, snapshot_auto_delete_behavior=volume.Volume.SnapshotAutoDeleteBehavior.DISABLED, snapshot_enabled=True, pod='pod_value', protocol=volume.Volume.Protocol.FIBRE_CHANNEL, boot_volume=True, performance_tier=common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED, notes='notes_value', workload_profile=volume.Volume.WorkloadProfile.GENERIC, instances=['instances_value'], attached=True)
        response = client.get_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.GetVolumeRequest()
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.storage_type == volume.Volume.StorageType.SSD
    assert response.state == volume.Volume.State.CREATING
    assert response.requested_size_gib == 1917
    assert response.originally_requested_size_gib == 3094
    assert response.current_size_gib == 1710
    assert response.emergency_size_gib == 1898
    assert response.max_size_gib == 1265
    assert response.auto_grown_size_gib == 2032
    assert response.remaining_space_gib == 1974
    assert response.snapshot_auto_delete_behavior == volume.Volume.SnapshotAutoDeleteBehavior.DISABLED
    assert response.snapshot_enabled is True
    assert response.pod == 'pod_value'
    assert response.protocol == volume.Volume.Protocol.FIBRE_CHANNEL
    assert response.boot_volume is True
    assert response.performance_tier == common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED
    assert response.notes == 'notes_value'
    assert response.workload_profile == volume.Volume.WorkloadProfile.GENERIC
    assert response.instances == ['instances_value']
    assert response.attached is True

def test_get_volume_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        client.get_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.GetVolumeRequest()

@pytest.mark.asyncio
async def test_get_volume_async(transport: str='grpc_asyncio', request_type=volume.GetVolumeRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume(name='name_value', id='id_value', storage_type=volume.Volume.StorageType.SSD, state=volume.Volume.State.CREATING, requested_size_gib=1917, originally_requested_size_gib=3094, current_size_gib=1710, emergency_size_gib=1898, max_size_gib=1265, auto_grown_size_gib=2032, remaining_space_gib=1974, snapshot_auto_delete_behavior=volume.Volume.SnapshotAutoDeleteBehavior.DISABLED, snapshot_enabled=True, pod='pod_value', protocol=volume.Volume.Protocol.FIBRE_CHANNEL, boot_volume=True, performance_tier=common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED, notes='notes_value', workload_profile=volume.Volume.WorkloadProfile.GENERIC, instances=['instances_value'], attached=True))
        response = await client.get_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.GetVolumeRequest()
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.storage_type == volume.Volume.StorageType.SSD
    assert response.state == volume.Volume.State.CREATING
    assert response.requested_size_gib == 1917
    assert response.originally_requested_size_gib == 3094
    assert response.current_size_gib == 1710
    assert response.emergency_size_gib == 1898
    assert response.max_size_gib == 1265
    assert response.auto_grown_size_gib == 2032
    assert response.remaining_space_gib == 1974
    assert response.snapshot_auto_delete_behavior == volume.Volume.SnapshotAutoDeleteBehavior.DISABLED
    assert response.snapshot_enabled is True
    assert response.pod == 'pod_value'
    assert response.protocol == volume.Volume.Protocol.FIBRE_CHANNEL
    assert response.boot_volume is True
    assert response.performance_tier == common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED
    assert response.notes == 'notes_value'
    assert response.workload_profile == volume.Volume.WorkloadProfile.GENERIC
    assert response.instances == ['instances_value']
    assert response.attached is True

@pytest.mark.asyncio
async def test_get_volume_async_from_dict():
    await test_get_volume_async(request_type=dict)

def test_get_volume_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.GetVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume()
        client.get_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_volume_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.GetVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume())
        await client.get_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_volume_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume()
        client.get_volume(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_volume_flattened_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_volume(volume.GetVolumeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_volume_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume())
        response = await client.get_volume(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_volume_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_volume(volume.GetVolumeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcb_volume.UpdateVolumeRequest, dict])
def test_update_volume(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume.UpdateVolumeRequest()
    assert isinstance(response, future.Future)

def test_update_volume_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        client.update_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume.UpdateVolumeRequest()

@pytest.mark.asyncio
async def test_update_volume_async(transport: str='grpc_asyncio', request_type=gcb_volume.UpdateVolumeRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume.UpdateVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_volume_async_from_dict():
    await test_update_volume_async(request_type=dict)

def test_update_volume_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume.UpdateVolumeRequest()
    request.volume.name = 'name_value'
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_volume_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume.UpdateVolumeRequest()
    request.volume.name = 'name_value'
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume.name=name_value') in kw['metadata']

def test_update_volume_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_volume(volume=gcb_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume
        mock_val = gcb_volume.Volume(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_volume_flattened_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_volume(gcb_volume.UpdateVolumeRequest(), volume=gcb_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_volume_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_volume(volume=gcb_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume
        mock_val = gcb_volume.Volume(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_volume_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_volume(gcb_volume.UpdateVolumeRequest(), volume=gcb_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [volume.RenameVolumeRequest, dict])
def test_rename_volume(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        call.return_value = volume.Volume(name='name_value', id='id_value', storage_type=volume.Volume.StorageType.SSD, state=volume.Volume.State.CREATING, requested_size_gib=1917, originally_requested_size_gib=3094, current_size_gib=1710, emergency_size_gib=1898, max_size_gib=1265, auto_grown_size_gib=2032, remaining_space_gib=1974, snapshot_auto_delete_behavior=volume.Volume.SnapshotAutoDeleteBehavior.DISABLED, snapshot_enabled=True, pod='pod_value', protocol=volume.Volume.Protocol.FIBRE_CHANNEL, boot_volume=True, performance_tier=common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED, notes='notes_value', workload_profile=volume.Volume.WorkloadProfile.GENERIC, instances=['instances_value'], attached=True)
        response = client.rename_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.RenameVolumeRequest()
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.storage_type == volume.Volume.StorageType.SSD
    assert response.state == volume.Volume.State.CREATING
    assert response.requested_size_gib == 1917
    assert response.originally_requested_size_gib == 3094
    assert response.current_size_gib == 1710
    assert response.emergency_size_gib == 1898
    assert response.max_size_gib == 1265
    assert response.auto_grown_size_gib == 2032
    assert response.remaining_space_gib == 1974
    assert response.snapshot_auto_delete_behavior == volume.Volume.SnapshotAutoDeleteBehavior.DISABLED
    assert response.snapshot_enabled is True
    assert response.pod == 'pod_value'
    assert response.protocol == volume.Volume.Protocol.FIBRE_CHANNEL
    assert response.boot_volume is True
    assert response.performance_tier == common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED
    assert response.notes == 'notes_value'
    assert response.workload_profile == volume.Volume.WorkloadProfile.GENERIC
    assert response.instances == ['instances_value']
    assert response.attached is True

def test_rename_volume_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        client.rename_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.RenameVolumeRequest()

@pytest.mark.asyncio
async def test_rename_volume_async(transport: str='grpc_asyncio', request_type=volume.RenameVolumeRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume(name='name_value', id='id_value', storage_type=volume.Volume.StorageType.SSD, state=volume.Volume.State.CREATING, requested_size_gib=1917, originally_requested_size_gib=3094, current_size_gib=1710, emergency_size_gib=1898, max_size_gib=1265, auto_grown_size_gib=2032, remaining_space_gib=1974, snapshot_auto_delete_behavior=volume.Volume.SnapshotAutoDeleteBehavior.DISABLED, snapshot_enabled=True, pod='pod_value', protocol=volume.Volume.Protocol.FIBRE_CHANNEL, boot_volume=True, performance_tier=common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED, notes='notes_value', workload_profile=volume.Volume.WorkloadProfile.GENERIC, instances=['instances_value'], attached=True))
        response = await client.rename_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.RenameVolumeRequest()
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.storage_type == volume.Volume.StorageType.SSD
    assert response.state == volume.Volume.State.CREATING
    assert response.requested_size_gib == 1917
    assert response.originally_requested_size_gib == 3094
    assert response.current_size_gib == 1710
    assert response.emergency_size_gib == 1898
    assert response.max_size_gib == 1265
    assert response.auto_grown_size_gib == 2032
    assert response.remaining_space_gib == 1974
    assert response.snapshot_auto_delete_behavior == volume.Volume.SnapshotAutoDeleteBehavior.DISABLED
    assert response.snapshot_enabled is True
    assert response.pod == 'pod_value'
    assert response.protocol == volume.Volume.Protocol.FIBRE_CHANNEL
    assert response.boot_volume is True
    assert response.performance_tier == common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED
    assert response.notes == 'notes_value'
    assert response.workload_profile == volume.Volume.WorkloadProfile.GENERIC
    assert response.instances == ['instances_value']
    assert response.attached is True

@pytest.mark.asyncio
async def test_rename_volume_async_from_dict():
    await test_rename_volume_async(request_type=dict)

def test_rename_volume_field_headers():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.RenameVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        call.return_value = volume.Volume()
        client.rename_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_rename_volume_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.RenameVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume())
        await client.rename_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_rename_volume_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        call.return_value = volume.Volume()
        client.rename_volume(name='name_value', new_volume_id='new_volume_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_volume_id
        mock_val = 'new_volume_id_value'
        assert arg == mock_val

def test_rename_volume_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.rename_volume(volume.RenameVolumeRequest(), name='name_value', new_volume_id='new_volume_id_value')

@pytest.mark.asyncio
async def test_rename_volume_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_volume), '__call__') as call:
        call.return_value = volume.Volume()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume())
        response = await client.rename_volume(name='name_value', new_volume_id='new_volume_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_volume_id
        mock_val = 'new_volume_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_rename_volume_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.rename_volume(volume.RenameVolumeRequest(), name='name_value', new_volume_id='new_volume_id_value')

@pytest.mark.parametrize('request_type', [volume.EvictVolumeRequest, dict])
def test_evict_volume(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.evict_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.EvictVolumeRequest()
    assert isinstance(response, future.Future)

def test_evict_volume_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        client.evict_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.EvictVolumeRequest()

@pytest.mark.asyncio
async def test_evict_volume_async(transport: str='grpc_asyncio', request_type=volume.EvictVolumeRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.evict_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.EvictVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_evict_volume_async_from_dict():
    await test_evict_volume_async(request_type=dict)

def test_evict_volume_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.EvictVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.evict_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_evict_volume_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.EvictVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.evict_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_evict_volume_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.evict_volume(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_evict_volume_flattened_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.evict_volume(volume.EvictVolumeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_evict_volume_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.evict_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.evict_volume(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_evict_volume_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.evict_volume(volume.EvictVolumeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcb_volume.ResizeVolumeRequest, dict])
def test_resize_volume(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.resize_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume.ResizeVolumeRequest()
    assert isinstance(response, future.Future)

def test_resize_volume_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        client.resize_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume.ResizeVolumeRequest()

@pytest.mark.asyncio
async def test_resize_volume_async(transport: str='grpc_asyncio', request_type=gcb_volume.ResizeVolumeRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.resize_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume.ResizeVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_resize_volume_async_from_dict():
    await test_resize_volume_async(request_type=dict)

def test_resize_volume_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume.ResizeVolumeRequest()
    request.volume = 'volume_value'
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.resize_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume=volume_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resize_volume_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume.ResizeVolumeRequest()
    request.volume = 'volume_value'
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.resize_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume=volume_value') in kw['metadata']

def test_resize_volume_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.resize_volume(volume='volume_value', size_gib=844)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume
        mock_val = 'volume_value'
        assert arg == mock_val
        arg = args[0].size_gib
        mock_val = 844
        assert arg == mock_val

def test_resize_volume_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resize_volume(gcb_volume.ResizeVolumeRequest(), volume='volume_value', size_gib=844)

@pytest.mark.asyncio
async def test_resize_volume_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resize_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.resize_volume(volume='volume_value', size_gib=844)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume
        mock_val = 'volume_value'
        assert arg == mock_val
        arg = args[0].size_gib
        mock_val = 844
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resize_volume_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resize_volume(gcb_volume.ResizeVolumeRequest(), volume='volume_value', size_gib=844)

@pytest.mark.parametrize('request_type', [network.ListNetworksRequest, dict])
def test_list_networks(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = network.ListNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_networks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.ListNetworksRequest()
    assert isinstance(response, pagers.ListNetworksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_networks_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        client.list_networks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.ListNetworksRequest()

@pytest.mark.asyncio
async def test_list_networks_async(transport: str='grpc_asyncio', request_type=network.ListNetworksRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.ListNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_networks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.ListNetworksRequest()
    assert isinstance(response, pagers.ListNetworksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_networks_async_from_dict():
    await test_list_networks_async(request_type=dict)

def test_list_networks_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.ListNetworksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = network.ListNetworksResponse()
        client.list_networks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_networks_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.ListNetworksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.ListNetworksResponse())
        await client.list_networks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_networks_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = network.ListNetworksResponse()
        client.list_networks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_networks_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_networks(network.ListNetworksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_networks_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = network.ListNetworksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.ListNetworksResponse())
        response = await client.list_networks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_networks_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_networks(network.ListNetworksRequest(), parent='parent_value')

def test_list_networks_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.side_effect = (network.ListNetworksResponse(networks=[network.Network(), network.Network(), network.Network()], next_page_token='abc'), network.ListNetworksResponse(networks=[], next_page_token='def'), network.ListNetworksResponse(networks=[network.Network()], next_page_token='ghi'), network.ListNetworksResponse(networks=[network.Network(), network.Network()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_networks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, network.Network) for i in results))

def test_list_networks_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.side_effect = (network.ListNetworksResponse(networks=[network.Network(), network.Network(), network.Network()], next_page_token='abc'), network.ListNetworksResponse(networks=[], next_page_token='def'), network.ListNetworksResponse(networks=[network.Network()], next_page_token='ghi'), network.ListNetworksResponse(networks=[network.Network(), network.Network()]), RuntimeError)
        pages = list(client.list_networks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_networks_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_networks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (network.ListNetworksResponse(networks=[network.Network(), network.Network(), network.Network()], next_page_token='abc'), network.ListNetworksResponse(networks=[], next_page_token='def'), network.ListNetworksResponse(networks=[network.Network()], next_page_token='ghi'), network.ListNetworksResponse(networks=[network.Network(), network.Network()]), RuntimeError)
        async_pager = await client.list_networks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, network.Network) for i in responses))

@pytest.mark.asyncio
async def test_list_networks_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_networks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (network.ListNetworksResponse(networks=[network.Network(), network.Network(), network.Network()], next_page_token='abc'), network.ListNetworksResponse(networks=[], next_page_token='def'), network.ListNetworksResponse(networks=[network.Network()], next_page_token='ghi'), network.ListNetworksResponse(networks=[network.Network(), network.Network()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_networks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [network.ListNetworkUsageRequest, dict])
def test_list_network_usage(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        call.return_value = network.ListNetworkUsageResponse()
        response = client.list_network_usage(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.ListNetworkUsageRequest()
    assert isinstance(response, network.ListNetworkUsageResponse)

def test_list_network_usage_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        client.list_network_usage()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.ListNetworkUsageRequest()

@pytest.mark.asyncio
async def test_list_network_usage_async(transport: str='grpc_asyncio', request_type=network.ListNetworkUsageRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.ListNetworkUsageResponse())
        response = await client.list_network_usage(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.ListNetworkUsageRequest()
    assert isinstance(response, network.ListNetworkUsageResponse)

@pytest.mark.asyncio
async def test_list_network_usage_async_from_dict():
    await test_list_network_usage_async(request_type=dict)

def test_list_network_usage_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.ListNetworkUsageRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        call.return_value = network.ListNetworkUsageResponse()
        client.list_network_usage(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_network_usage_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.ListNetworkUsageRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.ListNetworkUsageResponse())
        await client.list_network_usage(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

def test_list_network_usage_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        call.return_value = network.ListNetworkUsageResponse()
        client.list_network_usage(location='location_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val

def test_list_network_usage_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_network_usage(network.ListNetworkUsageRequest(), location='location_value')

@pytest.mark.asyncio
async def test_list_network_usage_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_network_usage), '__call__') as call:
        call.return_value = network.ListNetworkUsageResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.ListNetworkUsageResponse())
        response = await client.list_network_usage(location='location_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_network_usage_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_network_usage(network.ListNetworkUsageRequest(), location='location_value')

@pytest.mark.parametrize('request_type', [network.GetNetworkRequest, dict])
def test_get_network(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = network.Network(name='name_value', id='id_value', type_=network.Network.Type.CLIENT, ip_address='ip_address_value', mac_address=['mac_address_value'], state=network.Network.State.PROVISIONING, vlan_id='vlan_id_value', cidr='cidr_value', services_cidr='services_cidr_value', pod='pod_value', jumbo_frames_enabled=True, gateway_ip='gateway_ip_value')
        response = client.get_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.GetNetworkRequest()
    assert isinstance(response, network.Network)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == network.Network.Type.CLIENT
    assert response.ip_address == 'ip_address_value'
    assert response.mac_address == ['mac_address_value']
    assert response.state == network.Network.State.PROVISIONING
    assert response.vlan_id == 'vlan_id_value'
    assert response.cidr == 'cidr_value'
    assert response.services_cidr == 'services_cidr_value'
    assert response.pod == 'pod_value'
    assert response.jumbo_frames_enabled is True
    assert response.gateway_ip == 'gateway_ip_value'

def test_get_network_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        client.get_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.GetNetworkRequest()

@pytest.mark.asyncio
async def test_get_network_async(transport: str='grpc_asyncio', request_type=network.GetNetworkRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.Network(name='name_value', id='id_value', type_=network.Network.Type.CLIENT, ip_address='ip_address_value', mac_address=['mac_address_value'], state=network.Network.State.PROVISIONING, vlan_id='vlan_id_value', cidr='cidr_value', services_cidr='services_cidr_value', pod='pod_value', jumbo_frames_enabled=True, gateway_ip='gateway_ip_value'))
        response = await client.get_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.GetNetworkRequest()
    assert isinstance(response, network.Network)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == network.Network.Type.CLIENT
    assert response.ip_address == 'ip_address_value'
    assert response.mac_address == ['mac_address_value']
    assert response.state == network.Network.State.PROVISIONING
    assert response.vlan_id == 'vlan_id_value'
    assert response.cidr == 'cidr_value'
    assert response.services_cidr == 'services_cidr_value'
    assert response.pod == 'pod_value'
    assert response.jumbo_frames_enabled is True
    assert response.gateway_ip == 'gateway_ip_value'

@pytest.mark.asyncio
async def test_get_network_async_from_dict():
    await test_get_network_async(request_type=dict)

def test_get_network_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.GetNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = network.Network()
        client.get_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_network_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.GetNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.Network())
        await client.get_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_network_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = network.Network()
        client.get_network(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_network_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_network(network.GetNetworkRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_network_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = network.Network()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.Network())
        response = await client.get_network(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_network_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_network(network.GetNetworkRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcb_network.UpdateNetworkRequest, dict])
def test_update_network(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_network.UpdateNetworkRequest()
    assert isinstance(response, future.Future)

def test_update_network_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        client.update_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_network.UpdateNetworkRequest()

@pytest.mark.asyncio
async def test_update_network_async(transport: str='grpc_asyncio', request_type=gcb_network.UpdateNetworkRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_network.UpdateNetworkRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_network_async_from_dict():
    await test_update_network_async(request_type=dict)

def test_update_network_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_network.UpdateNetworkRequest()
    request.network.name = 'name_value'
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'network.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_network_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_network.UpdateNetworkRequest()
    request.network.name = 'name_value'
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'network.name=name_value') in kw['metadata']

def test_update_network_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_network(network=gcb_network.Network(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].network
        mock_val = gcb_network.Network(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_network_flattened_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_network(gcb_network.UpdateNetworkRequest(), network=gcb_network.Network(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_network_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_network(network=gcb_network.Network(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].network
        mock_val = gcb_network.Network(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_network_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_network(gcb_network.UpdateNetworkRequest(), network=gcb_network.Network(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [gcb_volume_snapshot.CreateVolumeSnapshotRequest, dict])
def test_create_volume_snapshot(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        call.return_value = gcb_volume_snapshot.VolumeSnapshot(name='name_value', id='id_value', description='description_value', storage_volume='storage_volume_value', type_=gcb_volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC)
        response = client.create_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume_snapshot.CreateVolumeSnapshotRequest()
    assert isinstance(response, gcb_volume_snapshot.VolumeSnapshot)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.description == 'description_value'
    assert response.storage_volume == 'storage_volume_value'
    assert response.type_ == gcb_volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC

def test_create_volume_snapshot_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        client.create_volume_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume_snapshot.CreateVolumeSnapshotRequest()

@pytest.mark.asyncio
async def test_create_volume_snapshot_async(transport: str='grpc_asyncio', request_type=gcb_volume_snapshot.CreateVolumeSnapshotRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcb_volume_snapshot.VolumeSnapshot(name='name_value', id='id_value', description='description_value', storage_volume='storage_volume_value', type_=gcb_volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC))
        response = await client.create_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume_snapshot.CreateVolumeSnapshotRequest()
    assert isinstance(response, gcb_volume_snapshot.VolumeSnapshot)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.description == 'description_value'
    assert response.storage_volume == 'storage_volume_value'
    assert response.type_ == gcb_volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC

@pytest.mark.asyncio
async def test_create_volume_snapshot_async_from_dict():
    await test_create_volume_snapshot_async(request_type=dict)

def test_create_volume_snapshot_field_headers():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume_snapshot.CreateVolumeSnapshotRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        call.return_value = gcb_volume_snapshot.VolumeSnapshot()
        client.create_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_volume_snapshot_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume_snapshot.CreateVolumeSnapshotRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcb_volume_snapshot.VolumeSnapshot())
        await client.create_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_volume_snapshot_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        call.return_value = gcb_volume_snapshot.VolumeSnapshot()
        client.create_volume_snapshot(parent='parent_value', volume_snapshot=gcb_volume_snapshot.VolumeSnapshot(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].volume_snapshot
        mock_val = gcb_volume_snapshot.VolumeSnapshot(name='name_value')
        assert arg == mock_val

def test_create_volume_snapshot_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_volume_snapshot(gcb_volume_snapshot.CreateVolumeSnapshotRequest(), parent='parent_value', volume_snapshot=gcb_volume_snapshot.VolumeSnapshot(name='name_value'))

@pytest.mark.asyncio
async def test_create_volume_snapshot_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_volume_snapshot), '__call__') as call:
        call.return_value = gcb_volume_snapshot.VolumeSnapshot()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcb_volume_snapshot.VolumeSnapshot())
        response = await client.create_volume_snapshot(parent='parent_value', volume_snapshot=gcb_volume_snapshot.VolumeSnapshot(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].volume_snapshot
        mock_val = gcb_volume_snapshot.VolumeSnapshot(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_volume_snapshot_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_volume_snapshot(gcb_volume_snapshot.CreateVolumeSnapshotRequest(), parent='parent_value', volume_snapshot=gcb_volume_snapshot.VolumeSnapshot(name='name_value'))

@pytest.mark.parametrize('request_type', [gcb_volume_snapshot.RestoreVolumeSnapshotRequest, dict])
def test_restore_volume_snapshot(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.restore_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume_snapshot.RestoreVolumeSnapshotRequest()
    assert isinstance(response, future.Future)

def test_restore_volume_snapshot_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        client.restore_volume_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume_snapshot.RestoreVolumeSnapshotRequest()

@pytest.mark.asyncio
async def test_restore_volume_snapshot_async(transport: str='grpc_asyncio', request_type=gcb_volume_snapshot.RestoreVolumeSnapshotRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restore_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_volume_snapshot.RestoreVolumeSnapshotRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_restore_volume_snapshot_async_from_dict():
    await test_restore_volume_snapshot_async(request_type=dict)

def test_restore_volume_snapshot_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume_snapshot.RestoreVolumeSnapshotRequest()
    request.volume_snapshot = 'volume_snapshot_value'
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restore_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume_snapshot=volume_snapshot_value') in kw['metadata']

@pytest.mark.asyncio
async def test_restore_volume_snapshot_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_volume_snapshot.RestoreVolumeSnapshotRequest()
    request.volume_snapshot = 'volume_snapshot_value'
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.restore_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume_snapshot=volume_snapshot_value') in kw['metadata']

def test_restore_volume_snapshot_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restore_volume_snapshot(volume_snapshot='volume_snapshot_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume_snapshot
        mock_val = 'volume_snapshot_value'
        assert arg == mock_val

def test_restore_volume_snapshot_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.restore_volume_snapshot(gcb_volume_snapshot.RestoreVolumeSnapshotRequest(), volume_snapshot='volume_snapshot_value')

@pytest.mark.asyncio
async def test_restore_volume_snapshot_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restore_volume_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restore_volume_snapshot(volume_snapshot='volume_snapshot_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume_snapshot
        mock_val = 'volume_snapshot_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_restore_volume_snapshot_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.restore_volume_snapshot(gcb_volume_snapshot.RestoreVolumeSnapshotRequest(), volume_snapshot='volume_snapshot_value')

@pytest.mark.parametrize('request_type', [volume_snapshot.DeleteVolumeSnapshotRequest, dict])
def test_delete_volume_snapshot(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        call.return_value = None
        response = client.delete_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.DeleteVolumeSnapshotRequest()
    assert response is None

def test_delete_volume_snapshot_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        client.delete_volume_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.DeleteVolumeSnapshotRequest()

@pytest.mark.asyncio
async def test_delete_volume_snapshot_async(transport: str='grpc_asyncio', request_type=volume_snapshot.DeleteVolumeSnapshotRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.DeleteVolumeSnapshotRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_volume_snapshot_async_from_dict():
    await test_delete_volume_snapshot_async(request_type=dict)

def test_delete_volume_snapshot_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume_snapshot.DeleteVolumeSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        call.return_value = None
        client.delete_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_volume_snapshot_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume_snapshot.DeleteVolumeSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_volume_snapshot_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        call.return_value = None
        client.delete_volume_snapshot(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_volume_snapshot_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_volume_snapshot(volume_snapshot.DeleteVolumeSnapshotRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_volume_snapshot_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_volume_snapshot), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_volume_snapshot(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_volume_snapshot_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_volume_snapshot(volume_snapshot.DeleteVolumeSnapshotRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [volume_snapshot.GetVolumeSnapshotRequest, dict])
def test_get_volume_snapshot(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        call.return_value = volume_snapshot.VolumeSnapshot(name='name_value', id='id_value', description='description_value', storage_volume='storage_volume_value', type_=volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC)
        response = client.get_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.GetVolumeSnapshotRequest()
    assert isinstance(response, volume_snapshot.VolumeSnapshot)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.description == 'description_value'
    assert response.storage_volume == 'storage_volume_value'
    assert response.type_ == volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC

def test_get_volume_snapshot_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        client.get_volume_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.GetVolumeSnapshotRequest()

@pytest.mark.asyncio
async def test_get_volume_snapshot_async(transport: str='grpc_asyncio', request_type=volume_snapshot.GetVolumeSnapshotRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume_snapshot.VolumeSnapshot(name='name_value', id='id_value', description='description_value', storage_volume='storage_volume_value', type_=volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC))
        response = await client.get_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.GetVolumeSnapshotRequest()
    assert isinstance(response, volume_snapshot.VolumeSnapshot)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.description == 'description_value'
    assert response.storage_volume == 'storage_volume_value'
    assert response.type_ == volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC

@pytest.mark.asyncio
async def test_get_volume_snapshot_async_from_dict():
    await test_get_volume_snapshot_async(request_type=dict)

def test_get_volume_snapshot_field_headers():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume_snapshot.GetVolumeSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        call.return_value = volume_snapshot.VolumeSnapshot()
        client.get_volume_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_volume_snapshot_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume_snapshot.GetVolumeSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume_snapshot.VolumeSnapshot())
        await client.get_volume_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_volume_snapshot_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        call.return_value = volume_snapshot.VolumeSnapshot()
        client.get_volume_snapshot(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_volume_snapshot_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_volume_snapshot(volume_snapshot.GetVolumeSnapshotRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_volume_snapshot_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_volume_snapshot), '__call__') as call:
        call.return_value = volume_snapshot.VolumeSnapshot()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume_snapshot.VolumeSnapshot())
        response = await client.get_volume_snapshot(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_volume_snapshot_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_volume_snapshot(volume_snapshot.GetVolumeSnapshotRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [volume_snapshot.ListVolumeSnapshotsRequest, dict])
def test_list_volume_snapshots(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.return_value = volume_snapshot.ListVolumeSnapshotsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_volume_snapshots(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.ListVolumeSnapshotsRequest()
    assert isinstance(response, pagers.ListVolumeSnapshotsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_volume_snapshots_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        client.list_volume_snapshots()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.ListVolumeSnapshotsRequest()

@pytest.mark.asyncio
async def test_list_volume_snapshots_async(transport: str='grpc_asyncio', request_type=volume_snapshot.ListVolumeSnapshotsRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume_snapshot.ListVolumeSnapshotsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_volume_snapshots(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume_snapshot.ListVolumeSnapshotsRequest()
    assert isinstance(response, pagers.ListVolumeSnapshotsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_volume_snapshots_async_from_dict():
    await test_list_volume_snapshots_async(request_type=dict)

def test_list_volume_snapshots_field_headers():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume_snapshot.ListVolumeSnapshotsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.return_value = volume_snapshot.ListVolumeSnapshotsResponse()
        client.list_volume_snapshots(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_volume_snapshots_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume_snapshot.ListVolumeSnapshotsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume_snapshot.ListVolumeSnapshotsResponse())
        await client.list_volume_snapshots(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_volume_snapshots_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.return_value = volume_snapshot.ListVolumeSnapshotsResponse()
        client.list_volume_snapshots(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_volume_snapshots_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_volume_snapshots(volume_snapshot.ListVolumeSnapshotsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_volume_snapshots_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.return_value = volume_snapshot.ListVolumeSnapshotsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume_snapshot.ListVolumeSnapshotsResponse())
        response = await client.list_volume_snapshots(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_volume_snapshots_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_volume_snapshots(volume_snapshot.ListVolumeSnapshotsRequest(), parent='parent_value')

def test_list_volume_snapshots_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.side_effect = (volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()], next_page_token='abc'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[], next_page_token='def'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot()], next_page_token='ghi'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_volume_snapshots(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, volume_snapshot.VolumeSnapshot) for i in results))

def test_list_volume_snapshots_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__') as call:
        call.side_effect = (volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()], next_page_token='abc'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[], next_page_token='def'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot()], next_page_token='ghi'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()]), RuntimeError)
        pages = list(client.list_volume_snapshots(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_volume_snapshots_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()], next_page_token='abc'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[], next_page_token='def'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot()], next_page_token='ghi'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()]), RuntimeError)
        async_pager = await client.list_volume_snapshots(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, volume_snapshot.VolumeSnapshot) for i in responses))

@pytest.mark.asyncio
async def test_list_volume_snapshots_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_volume_snapshots), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()], next_page_token='abc'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[], next_page_token='def'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot()], next_page_token='ghi'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_volume_snapshots(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [lun.GetLunRequest, dict])
def test_get_lun(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        call.return_value = lun.Lun(name='name_value', id='id_value', state=lun.Lun.State.CREATING, size_gb=739, multiprotocol_type=lun.Lun.MultiprotocolType.LINUX, storage_volume='storage_volume_value', shareable=True, boot_lun=True, storage_type=lun.Lun.StorageType.SSD, wwid='wwid_value', instances=['instances_value'])
        response = client.get_lun(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.GetLunRequest()
    assert isinstance(response, lun.Lun)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.state == lun.Lun.State.CREATING
    assert response.size_gb == 739
    assert response.multiprotocol_type == lun.Lun.MultiprotocolType.LINUX
    assert response.storage_volume == 'storage_volume_value'
    assert response.shareable is True
    assert response.boot_lun is True
    assert response.storage_type == lun.Lun.StorageType.SSD
    assert response.wwid == 'wwid_value'
    assert response.instances == ['instances_value']

def test_get_lun_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        client.get_lun()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.GetLunRequest()

@pytest.mark.asyncio
async def test_get_lun_async(transport: str='grpc_asyncio', request_type=lun.GetLunRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(lun.Lun(name='name_value', id='id_value', state=lun.Lun.State.CREATING, size_gb=739, multiprotocol_type=lun.Lun.MultiprotocolType.LINUX, storage_volume='storage_volume_value', shareable=True, boot_lun=True, storage_type=lun.Lun.StorageType.SSD, wwid='wwid_value', instances=['instances_value']))
        response = await client.get_lun(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.GetLunRequest()
    assert isinstance(response, lun.Lun)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.state == lun.Lun.State.CREATING
    assert response.size_gb == 739
    assert response.multiprotocol_type == lun.Lun.MultiprotocolType.LINUX
    assert response.storage_volume == 'storage_volume_value'
    assert response.shareable is True
    assert response.boot_lun is True
    assert response.storage_type == lun.Lun.StorageType.SSD
    assert response.wwid == 'wwid_value'
    assert response.instances == ['instances_value']

@pytest.mark.asyncio
async def test_get_lun_async_from_dict():
    await test_get_lun_async(request_type=dict)

def test_get_lun_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = lun.GetLunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        call.return_value = lun.Lun()
        client.get_lun(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_lun_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = lun.GetLunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(lun.Lun())
        await client.get_lun(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_lun_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        call.return_value = lun.Lun()
        client.get_lun(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_lun_flattened_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_lun(lun.GetLunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_lun_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_lun), '__call__') as call:
        call.return_value = lun.Lun()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(lun.Lun())
        response = await client.get_lun(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_lun_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_lun(lun.GetLunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [lun.ListLunsRequest, dict])
def test_list_luns(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.return_value = lun.ListLunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_luns(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.ListLunsRequest()
    assert isinstance(response, pagers.ListLunsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_luns_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        client.list_luns()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.ListLunsRequest()

@pytest.mark.asyncio
async def test_list_luns_async(transport: str='grpc_asyncio', request_type=lun.ListLunsRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(lun.ListLunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_luns(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.ListLunsRequest()
    assert isinstance(response, pagers.ListLunsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_luns_async_from_dict():
    await test_list_luns_async(request_type=dict)

def test_list_luns_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = lun.ListLunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.return_value = lun.ListLunsResponse()
        client.list_luns(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_luns_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = lun.ListLunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(lun.ListLunsResponse())
        await client.list_luns(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_luns_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.return_value = lun.ListLunsResponse()
        client.list_luns(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_luns_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_luns(lun.ListLunsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_luns_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.return_value = lun.ListLunsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(lun.ListLunsResponse())
        response = await client.list_luns(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_luns_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_luns(lun.ListLunsRequest(), parent='parent_value')

def test_list_luns_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.side_effect = (lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun(), lun.Lun()], next_page_token='abc'), lun.ListLunsResponse(luns=[], next_page_token='def'), lun.ListLunsResponse(luns=[lun.Lun()], next_page_token='ghi'), lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_luns(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, lun.Lun) for i in results))

def test_list_luns_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_luns), '__call__') as call:
        call.side_effect = (lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun(), lun.Lun()], next_page_token='abc'), lun.ListLunsResponse(luns=[], next_page_token='def'), lun.ListLunsResponse(luns=[lun.Lun()], next_page_token='ghi'), lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun()]), RuntimeError)
        pages = list(client.list_luns(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_luns_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_luns), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun(), lun.Lun()], next_page_token='abc'), lun.ListLunsResponse(luns=[], next_page_token='def'), lun.ListLunsResponse(luns=[lun.Lun()], next_page_token='ghi'), lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun()]), RuntimeError)
        async_pager = await client.list_luns(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, lun.Lun) for i in responses))

@pytest.mark.asyncio
async def test_list_luns_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_luns), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun(), lun.Lun()], next_page_token='abc'), lun.ListLunsResponse(luns=[], next_page_token='def'), lun.ListLunsResponse(luns=[lun.Lun()], next_page_token='ghi'), lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_luns(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [lun.EvictLunRequest, dict])
def test_evict_lun(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.evict_lun(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.EvictLunRequest()
    assert isinstance(response, future.Future)

def test_evict_lun_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        client.evict_lun()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.EvictLunRequest()

@pytest.mark.asyncio
async def test_evict_lun_async(transport: str='grpc_asyncio', request_type=lun.EvictLunRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.evict_lun(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == lun.EvictLunRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_evict_lun_async_from_dict():
    await test_evict_lun_async(request_type=dict)

def test_evict_lun_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = lun.EvictLunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.evict_lun(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_evict_lun_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = lun.EvictLunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.evict_lun(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_evict_lun_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.evict_lun(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_evict_lun_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.evict_lun(lun.EvictLunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_evict_lun_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.evict_lun), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.evict_lun(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_evict_lun_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.evict_lun(lun.EvictLunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [nfs_share.GetNfsShareRequest, dict])
def test_get_nfs_share(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare(name='name_value', nfs_share_id='nfs_share_id_value', id='id_value', state=nfs_share.NfsShare.State.PROVISIONED, volume='volume_value', requested_size_gib=1917, storage_type=nfs_share.NfsShare.StorageType.SSD)
        response = client.get_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.GetNfsShareRequest()
    assert isinstance(response, nfs_share.NfsShare)
    assert response.name == 'name_value'
    assert response.nfs_share_id == 'nfs_share_id_value'
    assert response.id == 'id_value'
    assert response.state == nfs_share.NfsShare.State.PROVISIONED
    assert response.volume == 'volume_value'
    assert response.requested_size_gib == 1917
    assert response.storage_type == nfs_share.NfsShare.StorageType.SSD

def test_get_nfs_share_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        client.get_nfs_share()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.GetNfsShareRequest()

@pytest.mark.asyncio
async def test_get_nfs_share_async(transport: str='grpc_asyncio', request_type=nfs_share.GetNfsShareRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.NfsShare(name='name_value', nfs_share_id='nfs_share_id_value', id='id_value', state=nfs_share.NfsShare.State.PROVISIONED, volume='volume_value', requested_size_gib=1917, storage_type=nfs_share.NfsShare.StorageType.SSD))
        response = await client.get_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.GetNfsShareRequest()
    assert isinstance(response, nfs_share.NfsShare)
    assert response.name == 'name_value'
    assert response.nfs_share_id == 'nfs_share_id_value'
    assert response.id == 'id_value'
    assert response.state == nfs_share.NfsShare.State.PROVISIONED
    assert response.volume == 'volume_value'
    assert response.requested_size_gib == 1917
    assert response.storage_type == nfs_share.NfsShare.StorageType.SSD

@pytest.mark.asyncio
async def test_get_nfs_share_async_from_dict():
    await test_get_nfs_share_async(request_type=dict)

def test_get_nfs_share_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.GetNfsShareRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare()
        client.get_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_nfs_share_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.GetNfsShareRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.NfsShare())
        await client.get_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_nfs_share_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare()
        client.get_nfs_share(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_nfs_share_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_nfs_share(nfs_share.GetNfsShareRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_nfs_share_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.NfsShare())
        response = await client.get_nfs_share(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_nfs_share_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_nfs_share(nfs_share.GetNfsShareRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [nfs_share.ListNfsSharesRequest, dict])
def test_list_nfs_shares(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.return_value = nfs_share.ListNfsSharesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_nfs_shares(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.ListNfsSharesRequest()
    assert isinstance(response, pagers.ListNfsSharesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_nfs_shares_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        client.list_nfs_shares()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.ListNfsSharesRequest()

@pytest.mark.asyncio
async def test_list_nfs_shares_async(transport: str='grpc_asyncio', request_type=nfs_share.ListNfsSharesRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.ListNfsSharesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_nfs_shares(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.ListNfsSharesRequest()
    assert isinstance(response, pagers.ListNfsSharesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_nfs_shares_async_from_dict():
    await test_list_nfs_shares_async(request_type=dict)

def test_list_nfs_shares_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.ListNfsSharesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.return_value = nfs_share.ListNfsSharesResponse()
        client.list_nfs_shares(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_nfs_shares_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.ListNfsSharesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.ListNfsSharesResponse())
        await client.list_nfs_shares(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_nfs_shares_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.return_value = nfs_share.ListNfsSharesResponse()
        client.list_nfs_shares(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_nfs_shares_flattened_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_nfs_shares(nfs_share.ListNfsSharesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_nfs_shares_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.return_value = nfs_share.ListNfsSharesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.ListNfsSharesResponse())
        response = await client.list_nfs_shares(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_nfs_shares_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_nfs_shares(nfs_share.ListNfsSharesRequest(), parent='parent_value')

def test_list_nfs_shares_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.side_effect = (nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare(), nfs_share.NfsShare()], next_page_token='abc'), nfs_share.ListNfsSharesResponse(nfs_shares=[], next_page_token='def'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare()], next_page_token='ghi'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_nfs_shares(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, nfs_share.NfsShare) for i in results))

def test_list_nfs_shares_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__') as call:
        call.side_effect = (nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare(), nfs_share.NfsShare()], next_page_token='abc'), nfs_share.ListNfsSharesResponse(nfs_shares=[], next_page_token='def'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare()], next_page_token='ghi'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare()]), RuntimeError)
        pages = list(client.list_nfs_shares(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_nfs_shares_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare(), nfs_share.NfsShare()], next_page_token='abc'), nfs_share.ListNfsSharesResponse(nfs_shares=[], next_page_token='def'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare()], next_page_token='ghi'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare()]), RuntimeError)
        async_pager = await client.list_nfs_shares(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, nfs_share.NfsShare) for i in responses))

@pytest.mark.asyncio
async def test_list_nfs_shares_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_nfs_shares), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare(), nfs_share.NfsShare()], next_page_token='abc'), nfs_share.ListNfsSharesResponse(nfs_shares=[], next_page_token='def'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare()], next_page_token='ghi'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_nfs_shares(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcb_nfs_share.UpdateNfsShareRequest, dict])
def test_update_nfs_share(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_nfs_share.UpdateNfsShareRequest()
    assert isinstance(response, future.Future)

def test_update_nfs_share_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        client.update_nfs_share()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_nfs_share.UpdateNfsShareRequest()

@pytest.mark.asyncio
async def test_update_nfs_share_async(transport: str='grpc_asyncio', request_type=gcb_nfs_share.UpdateNfsShareRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_nfs_share.UpdateNfsShareRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_nfs_share_async_from_dict():
    await test_update_nfs_share_async(request_type=dict)

def test_update_nfs_share_field_headers():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_nfs_share.UpdateNfsShareRequest()
    request.nfs_share.name = 'name_value'
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'nfs_share.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_nfs_share_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_nfs_share.UpdateNfsShareRequest()
    request.nfs_share.name = 'name_value'
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'nfs_share.name=name_value') in kw['metadata']

def test_update_nfs_share_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_nfs_share(nfs_share=gcb_nfs_share.NfsShare(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].nfs_share
        mock_val = gcb_nfs_share.NfsShare(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_nfs_share_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_nfs_share(gcb_nfs_share.UpdateNfsShareRequest(), nfs_share=gcb_nfs_share.NfsShare(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_nfs_share_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_nfs_share(nfs_share=gcb_nfs_share.NfsShare(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].nfs_share
        mock_val = gcb_nfs_share.NfsShare(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_nfs_share_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_nfs_share(gcb_nfs_share.UpdateNfsShareRequest(), nfs_share=gcb_nfs_share.NfsShare(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [gcb_nfs_share.CreateNfsShareRequest, dict])
def test_create_nfs_share(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_nfs_share.CreateNfsShareRequest()
    assert isinstance(response, future.Future)

def test_create_nfs_share_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        client.create_nfs_share()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_nfs_share.CreateNfsShareRequest()

@pytest.mark.asyncio
async def test_create_nfs_share_async(transport: str='grpc_asyncio', request_type=gcb_nfs_share.CreateNfsShareRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcb_nfs_share.CreateNfsShareRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_nfs_share_async_from_dict():
    await test_create_nfs_share_async(request_type=dict)

def test_create_nfs_share_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_nfs_share.CreateNfsShareRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_nfs_share_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcb_nfs_share.CreateNfsShareRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_nfs_share_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_nfs_share(parent='parent_value', nfs_share=gcb_nfs_share.NfsShare(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].nfs_share
        mock_val = gcb_nfs_share.NfsShare(name='name_value')
        assert arg == mock_val

def test_create_nfs_share_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_nfs_share(gcb_nfs_share.CreateNfsShareRequest(), parent='parent_value', nfs_share=gcb_nfs_share.NfsShare(name='name_value'))

@pytest.mark.asyncio
async def test_create_nfs_share_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_nfs_share(parent='parent_value', nfs_share=gcb_nfs_share.NfsShare(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].nfs_share
        mock_val = gcb_nfs_share.NfsShare(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_nfs_share_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_nfs_share(gcb_nfs_share.CreateNfsShareRequest(), parent='parent_value', nfs_share=gcb_nfs_share.NfsShare(name='name_value'))

@pytest.mark.parametrize('request_type', [nfs_share.RenameNfsShareRequest, dict])
def test_rename_nfs_share(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare(name='name_value', nfs_share_id='nfs_share_id_value', id='id_value', state=nfs_share.NfsShare.State.PROVISIONED, volume='volume_value', requested_size_gib=1917, storage_type=nfs_share.NfsShare.StorageType.SSD)
        response = client.rename_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.RenameNfsShareRequest()
    assert isinstance(response, nfs_share.NfsShare)
    assert response.name == 'name_value'
    assert response.nfs_share_id == 'nfs_share_id_value'
    assert response.id == 'id_value'
    assert response.state == nfs_share.NfsShare.State.PROVISIONED
    assert response.volume == 'volume_value'
    assert response.requested_size_gib == 1917
    assert response.storage_type == nfs_share.NfsShare.StorageType.SSD

def test_rename_nfs_share_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        client.rename_nfs_share()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.RenameNfsShareRequest()

@pytest.mark.asyncio
async def test_rename_nfs_share_async(transport: str='grpc_asyncio', request_type=nfs_share.RenameNfsShareRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.NfsShare(name='name_value', nfs_share_id='nfs_share_id_value', id='id_value', state=nfs_share.NfsShare.State.PROVISIONED, volume='volume_value', requested_size_gib=1917, storage_type=nfs_share.NfsShare.StorageType.SSD))
        response = await client.rename_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.RenameNfsShareRequest()
    assert isinstance(response, nfs_share.NfsShare)
    assert response.name == 'name_value'
    assert response.nfs_share_id == 'nfs_share_id_value'
    assert response.id == 'id_value'
    assert response.state == nfs_share.NfsShare.State.PROVISIONED
    assert response.volume == 'volume_value'
    assert response.requested_size_gib == 1917
    assert response.storage_type == nfs_share.NfsShare.StorageType.SSD

@pytest.mark.asyncio
async def test_rename_nfs_share_async_from_dict():
    await test_rename_nfs_share_async(request_type=dict)

def test_rename_nfs_share_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.RenameNfsShareRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare()
        client.rename_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_rename_nfs_share_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.RenameNfsShareRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.NfsShare())
        await client.rename_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_rename_nfs_share_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare()
        client.rename_nfs_share(name='name_value', new_nfsshare_id='new_nfsshare_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_nfsshare_id
        mock_val = 'new_nfsshare_id_value'
        assert arg == mock_val

def test_rename_nfs_share_flattened_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.rename_nfs_share(nfs_share.RenameNfsShareRequest(), name='name_value', new_nfsshare_id='new_nfsshare_id_value')

@pytest.mark.asyncio
async def test_rename_nfs_share_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_nfs_share), '__call__') as call:
        call.return_value = nfs_share.NfsShare()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(nfs_share.NfsShare())
        response = await client.rename_nfs_share(name='name_value', new_nfsshare_id='new_nfsshare_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_nfsshare_id
        mock_val = 'new_nfsshare_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_rename_nfs_share_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.rename_nfs_share(nfs_share.RenameNfsShareRequest(), name='name_value', new_nfsshare_id='new_nfsshare_id_value')

@pytest.mark.parametrize('request_type', [nfs_share.DeleteNfsShareRequest, dict])
def test_delete_nfs_share(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.DeleteNfsShareRequest()
    assert isinstance(response, future.Future)

def test_delete_nfs_share_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        client.delete_nfs_share()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.DeleteNfsShareRequest()

@pytest.mark.asyncio
async def test_delete_nfs_share_async(transport: str='grpc_asyncio', request_type=nfs_share.DeleteNfsShareRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == nfs_share.DeleteNfsShareRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_nfs_share_async_from_dict():
    await test_delete_nfs_share_async(request_type=dict)

def test_delete_nfs_share_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.DeleteNfsShareRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_nfs_share(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_nfs_share_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = nfs_share.DeleteNfsShareRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_nfs_share(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_nfs_share_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_nfs_share(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_nfs_share_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_nfs_share(nfs_share.DeleteNfsShareRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_nfs_share_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_nfs_share), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_nfs_share(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_nfs_share_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_nfs_share(nfs_share.DeleteNfsShareRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [provisioning.ListProvisioningQuotasRequest, dict])
def test_list_provisioning_quotas(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.return_value = provisioning.ListProvisioningQuotasResponse(next_page_token='next_page_token_value')
        response = client.list_provisioning_quotas(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.ListProvisioningQuotasRequest()
    assert isinstance(response, pagers.ListProvisioningQuotasPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_provisioning_quotas_empty_call():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        client.list_provisioning_quotas()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.ListProvisioningQuotasRequest()

@pytest.mark.asyncio
async def test_list_provisioning_quotas_async(transport: str='grpc_asyncio', request_type=provisioning.ListProvisioningQuotasRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ListProvisioningQuotasResponse(next_page_token='next_page_token_value'))
        response = await client.list_provisioning_quotas(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.ListProvisioningQuotasRequest()
    assert isinstance(response, pagers.ListProvisioningQuotasAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_provisioning_quotas_async_from_dict():
    await test_list_provisioning_quotas_async(request_type=dict)

def test_list_provisioning_quotas_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.ListProvisioningQuotasRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.return_value = provisioning.ListProvisioningQuotasResponse()
        client.list_provisioning_quotas(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_provisioning_quotas_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.ListProvisioningQuotasRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ListProvisioningQuotasResponse())
        await client.list_provisioning_quotas(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_provisioning_quotas_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.return_value = provisioning.ListProvisioningQuotasResponse()
        client.list_provisioning_quotas(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_provisioning_quotas_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_provisioning_quotas(provisioning.ListProvisioningQuotasRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_provisioning_quotas_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.return_value = provisioning.ListProvisioningQuotasResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ListProvisioningQuotasResponse())
        response = await client.list_provisioning_quotas(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_provisioning_quotas_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_provisioning_quotas(provisioning.ListProvisioningQuotasRequest(), parent='parent_value')

def test_list_provisioning_quotas_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.side_effect = (provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()], next_page_token='abc'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[], next_page_token='def'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota()], next_page_token='ghi'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_provisioning_quotas(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, provisioning.ProvisioningQuota) for i in results))

def test_list_provisioning_quotas_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__') as call:
        call.side_effect = (provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()], next_page_token='abc'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[], next_page_token='def'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota()], next_page_token='ghi'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()]), RuntimeError)
        pages = list(client.list_provisioning_quotas(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_provisioning_quotas_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()], next_page_token='abc'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[], next_page_token='def'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota()], next_page_token='ghi'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()]), RuntimeError)
        async_pager = await client.list_provisioning_quotas(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, provisioning.ProvisioningQuota) for i in responses))

@pytest.mark.asyncio
async def test_list_provisioning_quotas_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_provisioning_quotas), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()], next_page_token='abc'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[], next_page_token='def'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota()], next_page_token='ghi'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_provisioning_quotas(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [provisioning.SubmitProvisioningConfigRequest, dict])
def test_submit_provisioning_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        call.return_value = provisioning.SubmitProvisioningConfigResponse()
        response = client.submit_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.SubmitProvisioningConfigRequest()
    assert isinstance(response, provisioning.SubmitProvisioningConfigResponse)

def test_submit_provisioning_config_empty_call():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        client.submit_provisioning_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.SubmitProvisioningConfigRequest()

@pytest.mark.asyncio
async def test_submit_provisioning_config_async(transport: str='grpc_asyncio', request_type=provisioning.SubmitProvisioningConfigRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.SubmitProvisioningConfigResponse())
        response = await client.submit_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.SubmitProvisioningConfigRequest()
    assert isinstance(response, provisioning.SubmitProvisioningConfigResponse)

@pytest.mark.asyncio
async def test_submit_provisioning_config_async_from_dict():
    await test_submit_provisioning_config_async(request_type=dict)

def test_submit_provisioning_config_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.SubmitProvisioningConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        call.return_value = provisioning.SubmitProvisioningConfigResponse()
        client.submit_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_submit_provisioning_config_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.SubmitProvisioningConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.SubmitProvisioningConfigResponse())
        await client.submit_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_submit_provisioning_config_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        call.return_value = provisioning.SubmitProvisioningConfigResponse()
        client.submit_provisioning_config(parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].provisioning_config
        mock_val = provisioning.ProvisioningConfig(name='name_value')
        assert arg == mock_val

def test_submit_provisioning_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.submit_provisioning_config(provisioning.SubmitProvisioningConfigRequest(), parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))

@pytest.mark.asyncio
async def test_submit_provisioning_config_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_provisioning_config), '__call__') as call:
        call.return_value = provisioning.SubmitProvisioningConfigResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.SubmitProvisioningConfigResponse())
        response = await client.submit_provisioning_config(parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].provisioning_config
        mock_val = provisioning.ProvisioningConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_submit_provisioning_config_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.submit_provisioning_config(provisioning.SubmitProvisioningConfigRequest(), parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [provisioning.GetProvisioningConfigRequest, dict])
def test_get_provisioning_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value')
        response = client.get_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.GetProvisioningConfigRequest()
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

def test_get_provisioning_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        client.get_provisioning_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.GetProvisioningConfigRequest()

@pytest.mark.asyncio
async def test_get_provisioning_config_async(transport: str='grpc_asyncio', request_type=provisioning.GetProvisioningConfigRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value'))
        response = await client.get_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.GetProvisioningConfigRequest()
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

@pytest.mark.asyncio
async def test_get_provisioning_config_async_from_dict():
    await test_get_provisioning_config_async(request_type=dict)

def test_get_provisioning_config_field_headers():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.GetProvisioningConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        client.get_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_provisioning_config_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.GetProvisioningConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig())
        await client.get_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_provisioning_config_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        client.get_provisioning_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_provisioning_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_provisioning_config(provisioning.GetProvisioningConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_provisioning_config_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig())
        response = await client.get_provisioning_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_provisioning_config_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_provisioning_config(provisioning.GetProvisioningConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [provisioning.CreateProvisioningConfigRequest, dict])
def test_create_provisioning_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value')
        response = client.create_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.CreateProvisioningConfigRequest()
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

def test_create_provisioning_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        client.create_provisioning_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.CreateProvisioningConfigRequest()

@pytest.mark.asyncio
async def test_create_provisioning_config_async(transport: str='grpc_asyncio', request_type=provisioning.CreateProvisioningConfigRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value'))
        response = await client.create_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.CreateProvisioningConfigRequest()
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

@pytest.mark.asyncio
async def test_create_provisioning_config_async_from_dict():
    await test_create_provisioning_config_async(request_type=dict)

def test_create_provisioning_config_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.CreateProvisioningConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        client.create_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_provisioning_config_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.CreateProvisioningConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig())
        await client.create_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_provisioning_config_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        client.create_provisioning_config(parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].provisioning_config
        mock_val = provisioning.ProvisioningConfig(name='name_value')
        assert arg == mock_val

def test_create_provisioning_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_provisioning_config(provisioning.CreateProvisioningConfigRequest(), parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))

@pytest.mark.asyncio
async def test_create_provisioning_config_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig())
        response = await client.create_provisioning_config(parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].provisioning_config
        mock_val = provisioning.ProvisioningConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_provisioning_config_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_provisioning_config(provisioning.CreateProvisioningConfigRequest(), parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [provisioning.UpdateProvisioningConfigRequest, dict])
def test_update_provisioning_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value')
        response = client.update_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.UpdateProvisioningConfigRequest()
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

def test_update_provisioning_config_empty_call():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        client.update_provisioning_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.UpdateProvisioningConfigRequest()

@pytest.mark.asyncio
async def test_update_provisioning_config_async(transport: str='grpc_asyncio', request_type=provisioning.UpdateProvisioningConfigRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value'))
        response = await client.update_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == provisioning.UpdateProvisioningConfigRequest()
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

@pytest.mark.asyncio
async def test_update_provisioning_config_async_from_dict():
    await test_update_provisioning_config_async(request_type=dict)

def test_update_provisioning_config_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.UpdateProvisioningConfigRequest()
    request.provisioning_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        client.update_provisioning_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'provisioning_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_provisioning_config_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = provisioning.UpdateProvisioningConfigRequest()
    request.provisioning_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig())
        await client.update_provisioning_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'provisioning_config.name=name_value') in kw['metadata']

def test_update_provisioning_config_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        client.update_provisioning_config(provisioning_config=provisioning.ProvisioningConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].provisioning_config
        mock_val = provisioning.ProvisioningConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_provisioning_config_flattened_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_provisioning_config(provisioning.UpdateProvisioningConfigRequest(), provisioning_config=provisioning.ProvisioningConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_provisioning_config_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_provisioning_config), '__call__') as call:
        call.return_value = provisioning.ProvisioningConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(provisioning.ProvisioningConfig())
        response = await client.update_provisioning_config(provisioning_config=provisioning.ProvisioningConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].provisioning_config
        mock_val = provisioning.ProvisioningConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_provisioning_config_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_provisioning_config(provisioning.UpdateProvisioningConfigRequest(), provisioning_config=provisioning.ProvisioningConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [network.RenameNetworkRequest, dict])
def test_rename_network(request_type, transport: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        call.return_value = network.Network(name='name_value', id='id_value', type_=network.Network.Type.CLIENT, ip_address='ip_address_value', mac_address=['mac_address_value'], state=network.Network.State.PROVISIONING, vlan_id='vlan_id_value', cidr='cidr_value', services_cidr='services_cidr_value', pod='pod_value', jumbo_frames_enabled=True, gateway_ip='gateway_ip_value')
        response = client.rename_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.RenameNetworkRequest()
    assert isinstance(response, network.Network)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == network.Network.Type.CLIENT
    assert response.ip_address == 'ip_address_value'
    assert response.mac_address == ['mac_address_value']
    assert response.state == network.Network.State.PROVISIONING
    assert response.vlan_id == 'vlan_id_value'
    assert response.cidr == 'cidr_value'
    assert response.services_cidr == 'services_cidr_value'
    assert response.pod == 'pod_value'
    assert response.jumbo_frames_enabled is True
    assert response.gateway_ip == 'gateway_ip_value'

def test_rename_network_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        client.rename_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.RenameNetworkRequest()

@pytest.mark.asyncio
async def test_rename_network_async(transport: str='grpc_asyncio', request_type=network.RenameNetworkRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.Network(name='name_value', id='id_value', type_=network.Network.Type.CLIENT, ip_address='ip_address_value', mac_address=['mac_address_value'], state=network.Network.State.PROVISIONING, vlan_id='vlan_id_value', cidr='cidr_value', services_cidr='services_cidr_value', pod='pod_value', jumbo_frames_enabled=True, gateway_ip='gateway_ip_value'))
        response = await client.rename_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == network.RenameNetworkRequest()
    assert isinstance(response, network.Network)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == network.Network.Type.CLIENT
    assert response.ip_address == 'ip_address_value'
    assert response.mac_address == ['mac_address_value']
    assert response.state == network.Network.State.PROVISIONING
    assert response.vlan_id == 'vlan_id_value'
    assert response.cidr == 'cidr_value'
    assert response.services_cidr == 'services_cidr_value'
    assert response.pod == 'pod_value'
    assert response.jumbo_frames_enabled is True
    assert response.gateway_ip == 'gateway_ip_value'

@pytest.mark.asyncio
async def test_rename_network_async_from_dict():
    await test_rename_network_async(request_type=dict)

def test_rename_network_field_headers():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.RenameNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        call.return_value = network.Network()
        client.rename_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_rename_network_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = network.RenameNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.Network())
        await client.rename_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_rename_network_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        call.return_value = network.Network()
        client.rename_network(name='name_value', new_network_id='new_network_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_network_id
        mock_val = 'new_network_id_value'
        assert arg == mock_val

def test_rename_network_flattened_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.rename_network(network.RenameNetworkRequest(), name='name_value', new_network_id='new_network_id_value')

@pytest.mark.asyncio
async def test_rename_network_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rename_network), '__call__') as call:
        call.return_value = network.Network()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(network.Network())
        response = await client.rename_network(name='name_value', new_network_id='new_network_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].new_network_id
        mock_val = 'new_network_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_rename_network_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.rename_network(network.RenameNetworkRequest(), name='name_value', new_network_id='new_network_id_value')

@pytest.mark.parametrize('request_type', [osimage.ListOSImagesRequest, dict])
def test_list_os_images(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.return_value = osimage.ListOSImagesResponse(next_page_token='next_page_token_value')
        response = client.list_os_images(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == osimage.ListOSImagesRequest()
    assert isinstance(response, pagers.ListOSImagesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_images_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        client.list_os_images()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == osimage.ListOSImagesRequest()

@pytest.mark.asyncio
async def test_list_os_images_async(transport: str='grpc_asyncio', request_type=osimage.ListOSImagesRequest):
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(osimage.ListOSImagesResponse(next_page_token='next_page_token_value'))
        response = await client.list_os_images(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == osimage.ListOSImagesRequest()
    assert isinstance(response, pagers.ListOSImagesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_os_images_async_from_dict():
    await test_list_os_images_async(request_type=dict)

def test_list_os_images_field_headers():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    request = osimage.ListOSImagesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.return_value = osimage.ListOSImagesResponse()
        client.list_os_images(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_os_images_field_headers_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = osimage.ListOSImagesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(osimage.ListOSImagesResponse())
        await client.list_os_images(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_os_images_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.return_value = osimage.ListOSImagesResponse()
        client.list_os_images(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_os_images_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_os_images(osimage.ListOSImagesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_os_images_flattened_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.return_value = osimage.ListOSImagesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(osimage.ListOSImagesResponse())
        response = await client.list_os_images(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_os_images_flattened_error_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_os_images(osimage.ListOSImagesRequest(), parent='parent_value')

def test_list_os_images_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.side_effect = (osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage(), osimage.OSImage()], next_page_token='abc'), osimage.ListOSImagesResponse(os_images=[], next_page_token='def'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage()], next_page_token='ghi'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_os_images(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, osimage.OSImage) for i in results))

def test_list_os_images_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_images), '__call__') as call:
        call.side_effect = (osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage(), osimage.OSImage()], next_page_token='abc'), osimage.ListOSImagesResponse(os_images=[], next_page_token='def'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage()], next_page_token='ghi'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage()]), RuntimeError)
        pages = list(client.list_os_images(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_os_images_async_pager():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_images), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage(), osimage.OSImage()], next_page_token='abc'), osimage.ListOSImagesResponse(os_images=[], next_page_token='def'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage()], next_page_token='ghi'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage()]), RuntimeError)
        async_pager = await client.list_os_images(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, osimage.OSImage) for i in responses))

@pytest.mark.asyncio
async def test_list_os_images_async_pages():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_images), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage(), osimage.OSImage()], next_page_token='abc'), osimage.ListOSImagesResponse(os_images=[], next_page_token='def'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage()], next_page_token='ghi'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_os_images(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [instance.ListInstancesRequest, dict])
def test_list_instances_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = instance.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_instances(request)
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_rest_required_fields(request_type=instance.ListInstancesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = instance.ListInstancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = instance.ListInstancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_instances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_instances_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_instances_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_instances') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.ListInstancesRequest.pb(instance.ListInstancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = instance.ListInstancesResponse.to_json(instance.ListInstancesResponse())
        request = instance.ListInstancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = instance.ListInstancesResponse()
        client.list_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_instances_rest_bad_request(transport: str='rest', request_type=instance.ListInstancesRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_instances(request)

def test_list_instances_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance.ListInstancesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = instance.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_list_instances_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_instances(instance.ListInstancesRequest(), parent='parent_value')

def test_list_instances_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance(), instance.Instance()], next_page_token='abc'), instance.ListInstancesResponse(instances=[], next_page_token='def'), instance.ListInstancesResponse(instances=[instance.Instance()], next_page_token='ghi'), instance.ListInstancesResponse(instances=[instance.Instance(), instance.Instance()]))
        response = response + response
        response = tuple((instance.ListInstancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_instances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, instance.Instance) for i in results))
        pages = list(client.list_instances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [instance.GetInstanceRequest, dict])
def test_get_instance_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance.Instance(name='name_value', id='id_value', machine_type='machine_type_value', state=instance.Instance.State.PROVISIONING, hyperthreading_enabled=True, interactive_serial_console_enabled=True, os_image='os_image_value', pod='pod_value', network_template='network_template_value', login_info='login_info_value', workload_profile=common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC, firmware_version='firmware_version_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = instance.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance(request)
    assert isinstance(response, instance.Instance)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.machine_type == 'machine_type_value'
    assert response.state == instance.Instance.State.PROVISIONING
    assert response.hyperthreading_enabled is True
    assert response.interactive_serial_console_enabled is True
    assert response.os_image == 'os_image_value'
    assert response.pod == 'pod_value'
    assert response.network_template == 'network_template_value'
    assert response.login_info == 'login_info_value'
    assert response.workload_profile == common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC
    assert response.firmware_version == 'firmware_version_value'

def test_get_instance_rest_required_fields(request_type=instance.GetInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = instance.Instance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = instance.Instance.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_instance') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.GetInstanceRequest.pb(instance.GetInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = instance.Instance.to_json(instance.Instance())
        request = instance.GetInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = instance.Instance()
        client.get_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_rest_bad_request(transport: str='rest', request_type=instance.GetInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_instance(request)

def test_get_instance_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance.Instance()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = instance.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_get_instance_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance(instance.GetInstanceRequest(), name='name_value')

def test_get_instance_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_instance.UpdateInstanceRequest, dict])
def test_update_instance_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'instance': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
    request_init['instance'] = {'name': 'projects/sample1/locations/sample2/instances/sample3', 'id': 'id_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'machine_type': 'machine_type_value', 'state': 1, 'hyperthreading_enabled': True, 'labels': {}, 'luns': [{'name': 'name_value', 'id': 'id_value', 'state': 1, 'size_gb': 739, 'multiprotocol_type': 1, 'storage_volume': 'storage_volume_value', 'shareable': True, 'boot_lun': True, 'storage_type': 1, 'wwid': 'wwid_value', 'expire_time': {}, 'instances': ['instances_value1', 'instances_value2']}], 'volumes': [{'name': 'name_value', 'id': 'id_value', 'storage_type': 1, 'state': 1, 'requested_size_gib': 1917, 'originally_requested_size_gib': 3094, 'current_size_gib': 1710, 'emergency_size_gib': 1898, 'max_size_gib': 1265, 'auto_grown_size_gib': 2032, 'remaining_space_gib': 1974, 'snapshot_reservation_detail': {'reserved_space_gib': 1884, 'reserved_space_used_percent': 2859, 'reserved_space_remaining_gib': 2933, 'reserved_space_percent': 2331}, 'snapshot_auto_delete_behavior': 1, 'labels': {}, 'snapshot_enabled': True, 'pod': 'pod_value', 'protocol': 1, 'boot_volume': True, 'performance_tier': 1, 'notes': 'notes_value', 'workload_profile': 1, 'expire_time': {}, 'instances': ['instances_value1', 'instances_value2'], 'attached': True}], 'networks': [{'name': 'name_value', 'id': 'id_value', 'type_': 1, 'ip_address': 'ip_address_value', 'mac_address': ['mac_address_value1', 'mac_address_value2'], 'state': 1, 'vlan_id': 'vlan_id_value', 'cidr': 'cidr_value', 'vrf': {'name': 'name_value', 'state': 1, 'qos_policy': {'bandwidth_gbps': 0.1472}, 'vlan_attachments': [{'peer_vlan_id': 1256, 'peer_ip': 'peer_ip_value', 'router_ip': 'router_ip_value', 'pairing_key': 'pairing_key_value', 'qos_policy': {}, 'id': 'id_value', 'interconnect_attachment': 'interconnect_attachment_value'}]}, 'labels': {}, 'services_cidr': 'services_cidr_value', 'reservations': [{'start_address': 'start_address_value', 'end_address': 'end_address_value', 'note': 'note_value'}], 'pod': 'pod_value', 'mount_points': [{'instance': 'instance_value', 'logical_interface': 'logical_interface_value', 'default_gateway': True, 'ip_address': 'ip_address_value'}], 'jumbo_frames_enabled': True, 'gateway_ip': 'gateway_ip_value'}], 'interactive_serial_console_enabled': True, 'os_image': 'os_image_value', 'pod': 'pod_value', 'network_template': 'network_template_value', 'logical_interfaces': [{'logical_network_interfaces': [{'network': 'network_value', 'ip_address': 'ip_address_value', 'default_gateway': True, 'network_type': 1, 'id': 'id_value'}], 'name': 'name_value', 'interface_index': 1576}], 'login_info': 'login_info_value', 'workload_profile': 1, 'firmware_version': 'firmware_version_value'}
    test_field = gcb_instance.UpdateInstanceRequest.meta.fields['instance']

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
    for (field, value) in request_init['instance'].items():
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
                for i in range(0, len(request_init['instance'][field])):
                    del request_init['instance'][field][i][subfield]
            else:
                del request_init['instance'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_instance(request)
    assert response.operation.name == 'operations/spam'

def test_update_instance_rest_required_fields(request_type=gcb_instance.UpdateInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('instance',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_instance_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_update_instance') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_update_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_instance.UpdateInstanceRequest.pb(gcb_instance.UpdateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_instance.UpdateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_instance_rest_bad_request(transport: str='rest', request_type=gcb_instance.UpdateInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'instance': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_instance(request)

def test_update_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'instance': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
        mock_args = dict(instance=gcb_instance.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{instance.name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_update_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_instance(gcb_instance.UpdateInstanceRequest(), instance=gcb_instance.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_instance_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance.RenameInstanceRequest, dict])
def test_rename_instance_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance.Instance(name='name_value', id='id_value', machine_type='machine_type_value', state=instance.Instance.State.PROVISIONING, hyperthreading_enabled=True, interactive_serial_console_enabled=True, os_image='os_image_value', pod='pod_value', network_template='network_template_value', login_info='login_info_value', workload_profile=common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC, firmware_version='firmware_version_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = instance.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.rename_instance(request)
    assert isinstance(response, instance.Instance)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.machine_type == 'machine_type_value'
    assert response.state == instance.Instance.State.PROVISIONING
    assert response.hyperthreading_enabled is True
    assert response.interactive_serial_console_enabled is True
    assert response.os_image == 'os_image_value'
    assert response.pod == 'pod_value'
    assert response.network_template == 'network_template_value'
    assert response.login_info == 'login_info_value'
    assert response.workload_profile == common.WorkloadProfile.WORKLOAD_PROFILE_GENERIC
    assert response.firmware_version == 'firmware_version_value'

def test_rename_instance_rest_required_fields(request_type=instance.RenameInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['new_instance_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['newInstanceId'] = 'new_instance_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'newInstanceId' in jsonified_request
    assert jsonified_request['newInstanceId'] == 'new_instance_id_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = instance.Instance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = instance.Instance.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.rename_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_rename_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.rename_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'newInstanceId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_rename_instance_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_rename_instance') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_rename_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.RenameInstanceRequest.pb(instance.RenameInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = instance.Instance.to_json(instance.Instance())
        request = instance.RenameInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = instance.Instance()
        client.rename_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_rename_instance_rest_bad_request(transport: str='rest', request_type=instance.RenameInstanceRequest):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.rename_instance(request)

def test_rename_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance.Instance()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', new_instance_id='new_instance_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = instance.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.rename_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}:rename' % client.transport._host, args[1])

def test_rename_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.rename_instance(instance.RenameInstanceRequest(), name='name_value', new_instance_id='new_instance_id_value')

def test_rename_instance_rest_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance.ResetInstanceRequest, dict])
def test_reset_instance_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.reset_instance(request)
    assert response.operation.name == 'operations/spam'

def test_reset_instance_rest_required_fields(request_type=instance.ResetInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reset_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reset_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.reset_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_reset_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reset_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reset_instance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_reset_instance') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_reset_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.ResetInstanceRequest.pb(instance.ResetInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = instance.ResetInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.reset_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reset_instance_rest_bad_request(transport: str='rest', request_type=instance.ResetInstanceRequest):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reset_instance(request)

def test_reset_instance_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.reset_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}:reset' % client.transport._host, args[1])

def test_reset_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.reset_instance(instance.ResetInstanceRequest(), name='name_value')

def test_reset_instance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance.StartInstanceRequest, dict])
def test_start_instance_rest(request_type):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_instance(request)
    assert response.operation.name == 'operations/spam'

def test_start_instance_rest_required_fields(request_type=instance.StartInstanceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.start_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_start_instance_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.start_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_start_instance') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_start_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.StartInstanceRequest.pb(instance.StartInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = instance.StartInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.start_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_instance_rest_bad_request(transport: str='rest', request_type=instance.StartInstanceRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_instance(request)

def test_start_instance_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.start_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}:start' % client.transport._host, args[1])

def test_start_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.start_instance(instance.StartInstanceRequest(), name='name_value')

def test_start_instance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance.StopInstanceRequest, dict])
def test_stop_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_instance(request)
    assert response.operation.name == 'operations/spam'

def test_stop_instance_rest_required_fields(request_type=instance.StopInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_instance_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_instance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_stop_instance') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_stop_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.StopInstanceRequest.pb(instance.StopInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = instance.StopInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.stop_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_instance_rest_bad_request(transport: str='rest', request_type=instance.StopInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_instance(request)

def test_stop_instance_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}:stop' % client.transport._host, args[1])

def test_stop_instance_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_instance(instance.StopInstanceRequest(), name='name_value')

def test_stop_instance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance.EnableInteractiveSerialConsoleRequest, dict])
def test_enable_interactive_serial_console_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.enable_interactive_serial_console(request)
    assert response.operation.name == 'operations/spam'

def test_enable_interactive_serial_console_rest_required_fields(request_type=instance.EnableInteractiveSerialConsoleRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_interactive_serial_console._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_interactive_serial_console._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.enable_interactive_serial_console(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_enable_interactive_serial_console_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.enable_interactive_serial_console._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_enable_interactive_serial_console_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_enable_interactive_serial_console') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_enable_interactive_serial_console') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.EnableInteractiveSerialConsoleRequest.pb(instance.EnableInteractiveSerialConsoleRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = instance.EnableInteractiveSerialConsoleRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.enable_interactive_serial_console(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_enable_interactive_serial_console_rest_bad_request(transport: str='rest', request_type=instance.EnableInteractiveSerialConsoleRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.enable_interactive_serial_console(request)

def test_enable_interactive_serial_console_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.enable_interactive_serial_console(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}:enableInteractiveSerialConsole' % client.transport._host, args[1])

def test_enable_interactive_serial_console_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.enable_interactive_serial_console(instance.EnableInteractiveSerialConsoleRequest(), name='name_value')

def test_enable_interactive_serial_console_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance.DisableInteractiveSerialConsoleRequest, dict])
def test_disable_interactive_serial_console_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.disable_interactive_serial_console(request)
    assert response.operation.name == 'operations/spam'

def test_disable_interactive_serial_console_rest_required_fields(request_type=instance.DisableInteractiveSerialConsoleRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_interactive_serial_console._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_interactive_serial_console._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.disable_interactive_serial_console(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_disable_interactive_serial_console_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.disable_interactive_serial_console._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_disable_interactive_serial_console_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_disable_interactive_serial_console') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_disable_interactive_serial_console') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance.DisableInteractiveSerialConsoleRequest.pb(instance.DisableInteractiveSerialConsoleRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = instance.DisableInteractiveSerialConsoleRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.disable_interactive_serial_console(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_disable_interactive_serial_console_rest_bad_request(transport: str='rest', request_type=instance.DisableInteractiveSerialConsoleRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.disable_interactive_serial_console(request)

def test_disable_interactive_serial_console_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.disable_interactive_serial_console(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/instances/*}:disableInteractiveSerialConsole' % client.transport._host, args[1])

def test_disable_interactive_serial_console_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.disable_interactive_serial_console(instance.DisableInteractiveSerialConsoleRequest(), name='name_value')

def test_disable_interactive_serial_console_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_instance.DetachLunRequest, dict])
def test_detach_lun_rest(request_type):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.detach_lun(request)
    assert response.operation.name == 'operations/spam'

def test_detach_lun_rest_required_fields(request_type=gcb_instance.DetachLunRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['instance'] = ''
    request_init['lun'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).detach_lun._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instance'] = 'instance_value'
    jsonified_request['lun'] = 'lun_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).detach_lun._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instance' in jsonified_request
    assert jsonified_request['instance'] == 'instance_value'
    assert 'lun' in jsonified_request
    assert jsonified_request['lun'] == 'lun_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.detach_lun(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_detach_lun_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.detach_lun._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instance', 'lun'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_detach_lun_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_detach_lun') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_detach_lun') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_instance.DetachLunRequest.pb(gcb_instance.DetachLunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_instance.DetachLunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.detach_lun(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_detach_lun_rest_bad_request(transport: str='rest', request_type=gcb_instance.DetachLunRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.detach_lun(request)

def test_detach_lun_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(instance='instance_value', lun='lun_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.detach_lun(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{instance=projects/*/locations/*/instances/*}:detachLun' % client.transport._host, args[1])

def test_detach_lun_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.detach_lun(gcb_instance.DetachLunRequest(), instance='instance_value', lun='lun_value')

def test_detach_lun_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ssh_key.ListSSHKeysRequest, dict])
def test_list_ssh_keys_rest(request_type):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ssh_key.ListSSHKeysResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = ssh_key.ListSSHKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_ssh_keys(request)
    assert isinstance(response, pagers.ListSSHKeysPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_ssh_keys_rest_required_fields(request_type=ssh_key.ListSSHKeysRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_ssh_keys._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_ssh_keys._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ssh_key.ListSSHKeysResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ssh_key.ListSSHKeysResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_ssh_keys(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_ssh_keys_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_ssh_keys._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_ssh_keys_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_ssh_keys') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_ssh_keys') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ssh_key.ListSSHKeysRequest.pb(ssh_key.ListSSHKeysRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ssh_key.ListSSHKeysResponse.to_json(ssh_key.ListSSHKeysResponse())
        request = ssh_key.ListSSHKeysRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ssh_key.ListSSHKeysResponse()
        client.list_ssh_keys(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_ssh_keys_rest_bad_request(transport: str='rest', request_type=ssh_key.ListSSHKeysRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_ssh_keys(request)

def test_list_ssh_keys_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ssh_key.ListSSHKeysResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ssh_key.ListSSHKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_ssh_keys(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/sshKeys' % client.transport._host, args[1])

def test_list_ssh_keys_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_ssh_keys(ssh_key.ListSSHKeysRequest(), parent='parent_value')

def test_list_ssh_keys_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey(), ssh_key.SSHKey()], next_page_token='abc'), ssh_key.ListSSHKeysResponse(ssh_keys=[], next_page_token='def'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey()], next_page_token='ghi'), ssh_key.ListSSHKeysResponse(ssh_keys=[ssh_key.SSHKey(), ssh_key.SSHKey()]))
        response = response + response
        response = tuple((ssh_key.ListSSHKeysResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_ssh_keys(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, ssh_key.SSHKey) for i in results))
        pages = list(client.list_ssh_keys(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcb_ssh_key.CreateSSHKeyRequest, dict])
def test_create_ssh_key_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['ssh_key'] = {'name': 'name_value', 'public_key': 'public_key_value'}
    test_field = gcb_ssh_key.CreateSSHKeyRequest.meta.fields['ssh_key']

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
    for (field, value) in request_init['ssh_key'].items():
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
                for i in range(0, len(request_init['ssh_key'][field])):
                    del request_init['ssh_key'][field][i][subfield]
            else:
                del request_init['ssh_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcb_ssh_key.SSHKey(name='name_value', public_key='public_key_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcb_ssh_key.SSHKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_ssh_key(request)
    assert isinstance(response, gcb_ssh_key.SSHKey)
    assert response.name == 'name_value'
    assert response.public_key == 'public_key_value'

def test_create_ssh_key_rest_required_fields(request_type=gcb_ssh_key.CreateSSHKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['ssh_key_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'sshKeyId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_ssh_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'sshKeyId' in jsonified_request
    assert jsonified_request['sshKeyId'] == request_init['ssh_key_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['sshKeyId'] = 'ssh_key_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_ssh_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('ssh_key_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'sshKeyId' in jsonified_request
    assert jsonified_request['sshKeyId'] == 'ssh_key_id_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcb_ssh_key.SSHKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcb_ssh_key.SSHKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_ssh_key(request)
            expected_params = [('sshKeyId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_ssh_key_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_ssh_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('sshKeyId',)) & set(('parent', 'sshKey', 'sshKeyId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_ssh_key_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_create_ssh_key') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_create_ssh_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_ssh_key.CreateSSHKeyRequest.pb(gcb_ssh_key.CreateSSHKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcb_ssh_key.SSHKey.to_json(gcb_ssh_key.SSHKey())
        request = gcb_ssh_key.CreateSSHKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcb_ssh_key.SSHKey()
        client.create_ssh_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_ssh_key_rest_bad_request(transport: str='rest', request_type=gcb_ssh_key.CreateSSHKeyRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_ssh_key(request)

def test_create_ssh_key_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcb_ssh_key.SSHKey()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', ssh_key=gcb_ssh_key.SSHKey(name='name_value'), ssh_key_id='ssh_key_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcb_ssh_key.SSHKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_ssh_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/sshKeys' % client.transport._host, args[1])

def test_create_ssh_key_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_ssh_key(gcb_ssh_key.CreateSSHKeyRequest(), parent='parent_value', ssh_key=gcb_ssh_key.SSHKey(name='name_value'), ssh_key_id='ssh_key_id_value')

def test_create_ssh_key_rest_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ssh_key.DeleteSSHKeyRequest, dict])
def test_delete_ssh_key_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/sshKeys/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_ssh_key(request)
    assert response is None

def test_delete_ssh_key_rest_required_fields(request_type=ssh_key.DeleteSSHKeyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_ssh_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_ssh_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_ssh_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_ssh_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_ssh_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_ssh_key_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_delete_ssh_key') as pre:
        pre.assert_not_called()
        pb_message = ssh_key.DeleteSSHKeyRequest.pb(ssh_key.DeleteSSHKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = ssh_key.DeleteSSHKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_ssh_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_ssh_key_rest_bad_request(transport: str='rest', request_type=ssh_key.DeleteSSHKeyRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/sshKeys/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_ssh_key(request)

def test_delete_ssh_key_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/sshKeys/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_ssh_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/sshKeys/*}' % client.transport._host, args[1])

def test_delete_ssh_key_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_ssh_key(ssh_key.DeleteSSHKeyRequest(), name='name_value')

def test_delete_ssh_key_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume.ListVolumesRequest, dict])
def test_list_volumes_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.ListVolumesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.ListVolumesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_volumes(request)
    assert isinstance(response, pagers.ListVolumesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_volumes_rest_required_fields(request_type=volume.ListVolumesRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_volumes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_volumes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume.ListVolumesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume.ListVolumesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_volumes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_volumes_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_volumes._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_volumes_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_volumes') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_volumes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.ListVolumesRequest.pb(volume.ListVolumesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume.ListVolumesResponse.to_json(volume.ListVolumesResponse())
        request = volume.ListVolumesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume.ListVolumesResponse()
        client.list_volumes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_volumes_rest_bad_request(transport: str='rest', request_type=volume.ListVolumesRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_volumes(request)

def test_list_volumes_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.ListVolumesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.ListVolumesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_volumes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/volumes' % client.transport._host, args[1])

def test_list_volumes_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_volumes(volume.ListVolumesRequest(), parent='parent_value')

def test_list_volumes_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]))
        response = response + response
        response = tuple((volume.ListVolumesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_volumes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, volume.Volume) for i in results))
        pages = list(client.list_volumes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [volume.GetVolumeRequest, dict])
def test_get_volume_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.Volume(name='name_value', id='id_value', storage_type=volume.Volume.StorageType.SSD, state=volume.Volume.State.CREATING, requested_size_gib=1917, originally_requested_size_gib=3094, current_size_gib=1710, emergency_size_gib=1898, max_size_gib=1265, auto_grown_size_gib=2032, remaining_space_gib=1974, snapshot_auto_delete_behavior=volume.Volume.SnapshotAutoDeleteBehavior.DISABLED, snapshot_enabled=True, pod='pod_value', protocol=volume.Volume.Protocol.FIBRE_CHANNEL, boot_volume=True, performance_tier=common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED, notes='notes_value', workload_profile=volume.Volume.WorkloadProfile.GENERIC, instances=['instances_value'], attached=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.Volume.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_volume(request)
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.storage_type == volume.Volume.StorageType.SSD
    assert response.state == volume.Volume.State.CREATING
    assert response.requested_size_gib == 1917
    assert response.originally_requested_size_gib == 3094
    assert response.current_size_gib == 1710
    assert response.emergency_size_gib == 1898
    assert response.max_size_gib == 1265
    assert response.auto_grown_size_gib == 2032
    assert response.remaining_space_gib == 1974
    assert response.snapshot_auto_delete_behavior == volume.Volume.SnapshotAutoDeleteBehavior.DISABLED
    assert response.snapshot_enabled is True
    assert response.pod == 'pod_value'
    assert response.protocol == volume.Volume.Protocol.FIBRE_CHANNEL
    assert response.boot_volume is True
    assert response.performance_tier == common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED
    assert response.notes == 'notes_value'
    assert response.workload_profile == volume.Volume.WorkloadProfile.GENERIC
    assert response.instances == ['instances_value']
    assert response.attached is True

def test_get_volume_rest_required_fields(request_type=volume.GetVolumeRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume.Volume()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume.Volume.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_volume_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_volume_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_volume') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.GetVolumeRequest.pb(volume.GetVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume.Volume.to_json(volume.Volume())
        request = volume.GetVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume.Volume()
        client.get_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_volume_rest_bad_request(transport: str='rest', request_type=volume.GetVolumeRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_volume(request)

def test_get_volume_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.Volume()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.Volume.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*}' % client.transport._host, args[1])

def test_get_volume_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_volume(volume.GetVolumeRequest(), name='name_value')

def test_get_volume_rest_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_volume.UpdateVolumeRequest, dict])
def test_update_volume_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'volume': {'name': 'projects/sample1/locations/sample2/volumes/sample3'}}
    request_init['volume'] = {'name': 'projects/sample1/locations/sample2/volumes/sample3', 'id': 'id_value', 'storage_type': 1, 'state': 1, 'requested_size_gib': 1917, 'originally_requested_size_gib': 3094, 'current_size_gib': 1710, 'emergency_size_gib': 1898, 'max_size_gib': 1265, 'auto_grown_size_gib': 2032, 'remaining_space_gib': 1974, 'snapshot_reservation_detail': {'reserved_space_gib': 1884, 'reserved_space_used_percent': 2859, 'reserved_space_remaining_gib': 2933, 'reserved_space_percent': 2331}, 'snapshot_auto_delete_behavior': 1, 'labels': {}, 'snapshot_enabled': True, 'pod': 'pod_value', 'protocol': 1, 'boot_volume': True, 'performance_tier': 1, 'notes': 'notes_value', 'workload_profile': 1, 'expire_time': {'seconds': 751, 'nanos': 543}, 'instances': ['instances_value1', 'instances_value2'], 'attached': True}
    test_field = gcb_volume.UpdateVolumeRequest.meta.fields['volume']

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
    for (field, value) in request_init['volume'].items():
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
                for i in range(0, len(request_init['volume'][field])):
                    del request_init['volume'][field][i][subfield]
            else:
                del request_init['volume'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_volume(request)
    assert response.operation.name == 'operations/spam'

def test_update_volume_rest_required_fields(request_type=gcb_volume.UpdateVolumeRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_volume._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_volume_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('volume',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_volume_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_update_volume') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_update_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_volume.UpdateVolumeRequest.pb(gcb_volume.UpdateVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_volume.UpdateVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_volume_rest_bad_request(transport: str='rest', request_type=gcb_volume.UpdateVolumeRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'volume': {'name': 'projects/sample1/locations/sample2/volumes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_volume(request)

def test_update_volume_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'volume': {'name': 'projects/sample1/locations/sample2/volumes/sample3'}}
        mock_args = dict(volume=gcb_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{volume.name=projects/*/locations/*/volumes/*}' % client.transport._host, args[1])

def test_update_volume_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_volume(gcb_volume.UpdateVolumeRequest(), volume=gcb_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_volume_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume.RenameVolumeRequest, dict])
def test_rename_volume_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.Volume(name='name_value', id='id_value', storage_type=volume.Volume.StorageType.SSD, state=volume.Volume.State.CREATING, requested_size_gib=1917, originally_requested_size_gib=3094, current_size_gib=1710, emergency_size_gib=1898, max_size_gib=1265, auto_grown_size_gib=2032, remaining_space_gib=1974, snapshot_auto_delete_behavior=volume.Volume.SnapshotAutoDeleteBehavior.DISABLED, snapshot_enabled=True, pod='pod_value', protocol=volume.Volume.Protocol.FIBRE_CHANNEL, boot_volume=True, performance_tier=common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED, notes='notes_value', workload_profile=volume.Volume.WorkloadProfile.GENERIC, instances=['instances_value'], attached=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.Volume.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.rename_volume(request)
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.storage_type == volume.Volume.StorageType.SSD
    assert response.state == volume.Volume.State.CREATING
    assert response.requested_size_gib == 1917
    assert response.originally_requested_size_gib == 3094
    assert response.current_size_gib == 1710
    assert response.emergency_size_gib == 1898
    assert response.max_size_gib == 1265
    assert response.auto_grown_size_gib == 2032
    assert response.remaining_space_gib == 1974
    assert response.snapshot_auto_delete_behavior == volume.Volume.SnapshotAutoDeleteBehavior.DISABLED
    assert response.snapshot_enabled is True
    assert response.pod == 'pod_value'
    assert response.protocol == volume.Volume.Protocol.FIBRE_CHANNEL
    assert response.boot_volume is True
    assert response.performance_tier == common.VolumePerformanceTier.VOLUME_PERFORMANCE_TIER_SHARED
    assert response.notes == 'notes_value'
    assert response.workload_profile == volume.Volume.WorkloadProfile.GENERIC
    assert response.instances == ['instances_value']
    assert response.attached is True

def test_rename_volume_rest_required_fields(request_type=volume.RenameVolumeRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['new_volume_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['newVolumeId'] = 'new_volume_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'newVolumeId' in jsonified_request
    assert jsonified_request['newVolumeId'] == 'new_volume_id_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume.Volume()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume.Volume.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.rename_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_rename_volume_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.rename_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'newVolumeId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_rename_volume_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_rename_volume') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_rename_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.RenameVolumeRequest.pb(volume.RenameVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume.Volume.to_json(volume.Volume())
        request = volume.RenameVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume.Volume()
        client.rename_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_rename_volume_rest_bad_request(transport: str='rest', request_type=volume.RenameVolumeRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.rename_volume(request)

def test_rename_volume_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.Volume()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(name='name_value', new_volume_id='new_volume_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.Volume.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.rename_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*}:rename' % client.transport._host, args[1])

def test_rename_volume_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.rename_volume(volume.RenameVolumeRequest(), name='name_value', new_volume_id='new_volume_id_value')

def test_rename_volume_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume.EvictVolumeRequest, dict])
def test_evict_volume_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.evict_volume(request)
    assert response.operation.name == 'operations/spam'

def test_evict_volume_rest_required_fields(request_type=volume.EvictVolumeRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).evict_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).evict_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.evict_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_evict_volume_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.evict_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_evict_volume_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_evict_volume') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_evict_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.EvictVolumeRequest.pb(volume.EvictVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = volume.EvictVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.evict_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_evict_volume_rest_bad_request(transport: str='rest', request_type=volume.EvictVolumeRequest):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.evict_volume(request)

def test_evict_volume_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.evict_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*}:evict' % client.transport._host, args[1])

def test_evict_volume_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.evict_volume(volume.EvictVolumeRequest(), name='name_value')

def test_evict_volume_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_volume.ResizeVolumeRequest, dict])
def test_resize_volume_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'volume': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resize_volume(request)
    assert response.operation.name == 'operations/spam'

def test_resize_volume_rest_required_fields(request_type=gcb_volume.ResizeVolumeRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['volume'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['volume'] = 'volume_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'volume' in jsonified_request
    assert jsonified_request['volume'] == 'volume_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resize_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resize_volume_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resize_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('volume',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resize_volume_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_resize_volume') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_resize_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_volume.ResizeVolumeRequest.pb(gcb_volume.ResizeVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_volume.ResizeVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.resize_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resize_volume_rest_bad_request(transport: str='rest', request_type=gcb_volume.ResizeVolumeRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'volume': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resize_volume(request)

def test_resize_volume_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'volume': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(volume='volume_value', size_gib=844)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resize_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{volume=projects/*/locations/*/volumes/*}:resize' % client.transport._host, args[1])

def test_resize_volume_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resize_volume(gcb_volume.ResizeVolumeRequest(), volume='volume_value', size_gib=844)

def test_resize_volume_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [network.ListNetworksRequest, dict])
def test_list_networks_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.ListNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = network.ListNetworksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_networks(request)
    assert isinstance(response, pagers.ListNetworksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_networks_rest_required_fields(request_type=network.ListNetworksRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_networks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_networks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = network.ListNetworksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = network.ListNetworksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_networks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_networks_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_networks._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_networks_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_networks') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_networks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = network.ListNetworksRequest.pb(network.ListNetworksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = network.ListNetworksResponse.to_json(network.ListNetworksResponse())
        request = network.ListNetworksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = network.ListNetworksResponse()
        client.list_networks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_networks_rest_bad_request(transport: str='rest', request_type=network.ListNetworksRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_networks(request)

def test_list_networks_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.ListNetworksResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = network.ListNetworksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_networks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/networks' % client.transport._host, args[1])

def test_list_networks_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_networks(network.ListNetworksRequest(), parent='parent_value')

def test_list_networks_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (network.ListNetworksResponse(networks=[network.Network(), network.Network(), network.Network()], next_page_token='abc'), network.ListNetworksResponse(networks=[], next_page_token='def'), network.ListNetworksResponse(networks=[network.Network()], next_page_token='ghi'), network.ListNetworksResponse(networks=[network.Network(), network.Network()]))
        response = response + response
        response = tuple((network.ListNetworksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_networks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, network.Network) for i in results))
        pages = list(client.list_networks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [network.ListNetworkUsageRequest, dict])
def test_list_network_usage_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.ListNetworkUsageResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = network.ListNetworkUsageResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_network_usage(request)
    assert isinstance(response, network.ListNetworkUsageResponse)

def test_list_network_usage_rest_required_fields(request_type=network.ListNetworkUsageRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['location'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_network_usage._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['location'] = 'location_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_network_usage._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'location' in jsonified_request
    assert jsonified_request['location'] == 'location_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = network.ListNetworkUsageResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = network.ListNetworkUsageResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_network_usage(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_network_usage_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_network_usage._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('location',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_network_usage_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_network_usage') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_network_usage') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = network.ListNetworkUsageRequest.pb(network.ListNetworkUsageRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = network.ListNetworkUsageResponse.to_json(network.ListNetworkUsageResponse())
        request = network.ListNetworkUsageRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = network.ListNetworkUsageResponse()
        client.list_network_usage(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_network_usage_rest_bad_request(transport: str='rest', request_type=network.ListNetworkUsageRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_network_usage(request)

def test_list_network_usage_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.ListNetworkUsageResponse()
        sample_request = {'location': 'projects/sample1/locations/sample2'}
        mock_args = dict(location='location_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = network.ListNetworkUsageResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_network_usage(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{location=projects/*/locations/*}/networks:listNetworkUsage' % client.transport._host, args[1])

def test_list_network_usage_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_network_usage(network.ListNetworkUsageRequest(), location='location_value')

def test_list_network_usage_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [network.GetNetworkRequest, dict])
def test_get_network_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/networks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.Network(name='name_value', id='id_value', type_=network.Network.Type.CLIENT, ip_address='ip_address_value', mac_address=['mac_address_value'], state=network.Network.State.PROVISIONING, vlan_id='vlan_id_value', cidr='cidr_value', services_cidr='services_cidr_value', pod='pod_value', jumbo_frames_enabled=True, gateway_ip='gateway_ip_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = network.Network.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_network(request)
    assert isinstance(response, network.Network)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == network.Network.Type.CLIENT
    assert response.ip_address == 'ip_address_value'
    assert response.mac_address == ['mac_address_value']
    assert response.state == network.Network.State.PROVISIONING
    assert response.vlan_id == 'vlan_id_value'
    assert response.cidr == 'cidr_value'
    assert response.services_cidr == 'services_cidr_value'
    assert response.pod == 'pod_value'
    assert response.jumbo_frames_enabled is True
    assert response.gateway_ip == 'gateway_ip_value'

def test_get_network_rest_required_fields(request_type=network.GetNetworkRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = network.Network()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = network.Network.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_network_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_network._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_network_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_network') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = network.GetNetworkRequest.pb(network.GetNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = network.Network.to_json(network.Network())
        request = network.GetNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = network.Network()
        client.get_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_network_rest_bad_request(transport: str='rest', request_type=network.GetNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/networks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_network(request)

def test_get_network_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.Network()
        sample_request = {'name': 'projects/sample1/locations/sample2/networks/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = network.Network.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/networks/*}' % client.transport._host, args[1])

def test_get_network_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_network(network.GetNetworkRequest(), name='name_value')

def test_get_network_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_network.UpdateNetworkRequest, dict])
def test_update_network_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'network': {'name': 'projects/sample1/locations/sample2/networks/sample3'}}
    request_init['network'] = {'name': 'projects/sample1/locations/sample2/networks/sample3', 'id': 'id_value', 'type_': 1, 'ip_address': 'ip_address_value', 'mac_address': ['mac_address_value1', 'mac_address_value2'], 'state': 1, 'vlan_id': 'vlan_id_value', 'cidr': 'cidr_value', 'vrf': {'name': 'name_value', 'state': 1, 'qos_policy': {'bandwidth_gbps': 0.1472}, 'vlan_attachments': [{'peer_vlan_id': 1256, 'peer_ip': 'peer_ip_value', 'router_ip': 'router_ip_value', 'pairing_key': 'pairing_key_value', 'qos_policy': {}, 'id': 'id_value', 'interconnect_attachment': 'interconnect_attachment_value'}]}, 'labels': {}, 'services_cidr': 'services_cidr_value', 'reservations': [{'start_address': 'start_address_value', 'end_address': 'end_address_value', 'note': 'note_value'}], 'pod': 'pod_value', 'mount_points': [{'instance': 'instance_value', 'logical_interface': 'logical_interface_value', 'default_gateway': True, 'ip_address': 'ip_address_value'}], 'jumbo_frames_enabled': True, 'gateway_ip': 'gateway_ip_value'}
    test_field = gcb_network.UpdateNetworkRequest.meta.fields['network']

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
    for (field, value) in request_init['network'].items():
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
                for i in range(0, len(request_init['network'][field])):
                    del request_init['network'][field][i][subfield]
            else:
                del request_init['network'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_network(request)
    assert response.operation.name == 'operations/spam'

def test_update_network_rest_required_fields(request_type=gcb_network.UpdateNetworkRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_network._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_network_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_network._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('network',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_update_network') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_update_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_network.UpdateNetworkRequest.pb(gcb_network.UpdateNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_network.UpdateNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_network_rest_bad_request(transport: str='rest', request_type=gcb_network.UpdateNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'network': {'name': 'projects/sample1/locations/sample2/networks/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_network(request)

def test_update_network_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'network': {'name': 'projects/sample1/locations/sample2/networks/sample3'}}
        mock_args = dict(network=gcb_network.Network(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{network.name=projects/*/locations/*/networks/*}' % client.transport._host, args[1])

def test_update_network_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_network(gcb_network.UpdateNetworkRequest(), network=gcb_network.Network(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_network_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_volume_snapshot.CreateVolumeSnapshotRequest, dict])
def test_create_volume_snapshot_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request_init['volume_snapshot'] = {'name': 'name_value', 'id': 'id_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'storage_volume': 'storage_volume_value', 'type_': 1}
    test_field = gcb_volume_snapshot.CreateVolumeSnapshotRequest.meta.fields['volume_snapshot']

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
    for (field, value) in request_init['volume_snapshot'].items():
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
                for i in range(0, len(request_init['volume_snapshot'][field])):
                    del request_init['volume_snapshot'][field][i][subfield]
            else:
                del request_init['volume_snapshot'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcb_volume_snapshot.VolumeSnapshot(name='name_value', id='id_value', description='description_value', storage_volume='storage_volume_value', type_=gcb_volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcb_volume_snapshot.VolumeSnapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_volume_snapshot(request)
    assert isinstance(response, gcb_volume_snapshot.VolumeSnapshot)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.description == 'description_value'
    assert response.storage_volume == 'storage_volume_value'
    assert response.type_ == gcb_volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC

def test_create_volume_snapshot_rest_required_fields(request_type=gcb_volume_snapshot.CreateVolumeSnapshotRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcb_volume_snapshot.VolumeSnapshot()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcb_volume_snapshot.VolumeSnapshot.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_volume_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_volume_snapshot_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_volume_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'volumeSnapshot'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_volume_snapshot_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_create_volume_snapshot') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_create_volume_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_volume_snapshot.CreateVolumeSnapshotRequest.pb(gcb_volume_snapshot.CreateVolumeSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcb_volume_snapshot.VolumeSnapshot.to_json(gcb_volume_snapshot.VolumeSnapshot())
        request = gcb_volume_snapshot.CreateVolumeSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcb_volume_snapshot.VolumeSnapshot()
        client.create_volume_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_volume_snapshot_rest_bad_request(transport: str='rest', request_type=gcb_volume_snapshot.CreateVolumeSnapshotRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_volume_snapshot(request)

def test_create_volume_snapshot_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcb_volume_snapshot.VolumeSnapshot()
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value', volume_snapshot=gcb_volume_snapshot.VolumeSnapshot(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcb_volume_snapshot.VolumeSnapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_volume_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/volumes/*}/snapshots' % client.transport._host, args[1])

def test_create_volume_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_volume_snapshot(gcb_volume_snapshot.CreateVolumeSnapshotRequest(), parent='parent_value', volume_snapshot=gcb_volume_snapshot.VolumeSnapshot(name='name_value'))

def test_create_volume_snapshot_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_volume_snapshot.RestoreVolumeSnapshotRequest, dict])
def test_restore_volume_snapshot_rest(request_type):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'volume_snapshot': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.restore_volume_snapshot(request)
    assert response.operation.name == 'operations/spam'

def test_restore_volume_snapshot_rest_required_fields(request_type=gcb_volume_snapshot.RestoreVolumeSnapshotRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['volume_snapshot'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['volumeSnapshot'] = 'volume_snapshot_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'volumeSnapshot' in jsonified_request
    assert jsonified_request['volumeSnapshot'] == 'volume_snapshot_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.restore_volume_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_restore_volume_snapshot_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.restore_volume_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('volumeSnapshot',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_restore_volume_snapshot_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_restore_volume_snapshot') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_restore_volume_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_volume_snapshot.RestoreVolumeSnapshotRequest.pb(gcb_volume_snapshot.RestoreVolumeSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_volume_snapshot.RestoreVolumeSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.restore_volume_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_restore_volume_snapshot_rest_bad_request(transport: str='rest', request_type=gcb_volume_snapshot.RestoreVolumeSnapshotRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'volume_snapshot': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.restore_volume_snapshot(request)

def test_restore_volume_snapshot_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'volume_snapshot': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
        mock_args = dict(volume_snapshot='volume_snapshot_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.restore_volume_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{volume_snapshot=projects/*/locations/*/volumes/*/snapshots/*}:restoreVolumeSnapshot' % client.transport._host, args[1])

def test_restore_volume_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.restore_volume_snapshot(gcb_volume_snapshot.RestoreVolumeSnapshotRequest(), volume_snapshot='volume_snapshot_value')

def test_restore_volume_snapshot_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume_snapshot.DeleteVolumeSnapshotRequest, dict])
def test_delete_volume_snapshot_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_volume_snapshot(request)
    assert response is None

def test_delete_volume_snapshot_rest_required_fields(request_type=volume_snapshot.DeleteVolumeSnapshotRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_volume_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_volume_snapshot_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_volume_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_volume_snapshot_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_delete_volume_snapshot') as pre:
        pre.assert_not_called()
        pb_message = volume_snapshot.DeleteVolumeSnapshotRequest.pb(volume_snapshot.DeleteVolumeSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = volume_snapshot.DeleteVolumeSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_volume_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_volume_snapshot_rest_bad_request(transport: str='rest', request_type=volume_snapshot.DeleteVolumeSnapshotRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_volume_snapshot(request)

def test_delete_volume_snapshot_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_volume_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*/snapshots/*}' % client.transport._host, args[1])

def test_delete_volume_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_volume_snapshot(volume_snapshot.DeleteVolumeSnapshotRequest(), name='name_value')

def test_delete_volume_snapshot_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume_snapshot.GetVolumeSnapshotRequest, dict])
def test_get_volume_snapshot_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume_snapshot.VolumeSnapshot(name='name_value', id='id_value', description='description_value', storage_volume='storage_volume_value', type_=volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume_snapshot.VolumeSnapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_volume_snapshot(request)
    assert isinstance(response, volume_snapshot.VolumeSnapshot)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.description == 'description_value'
    assert response.storage_volume == 'storage_volume_value'
    assert response.type_ == volume_snapshot.VolumeSnapshot.SnapshotType.AD_HOC

def test_get_volume_snapshot_rest_required_fields(request_type=volume_snapshot.GetVolumeSnapshotRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_volume_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume_snapshot.VolumeSnapshot()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume_snapshot.VolumeSnapshot.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_volume_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_volume_snapshot_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_volume_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_volume_snapshot_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_volume_snapshot') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_volume_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume_snapshot.GetVolumeSnapshotRequest.pb(volume_snapshot.GetVolumeSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume_snapshot.VolumeSnapshot.to_json(volume_snapshot.VolumeSnapshot())
        request = volume_snapshot.GetVolumeSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume_snapshot.VolumeSnapshot()
        client.get_volume_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_volume_snapshot_rest_bad_request(transport: str='rest', request_type=volume_snapshot.GetVolumeSnapshotRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_volume_snapshot(request)

def test_get_volume_snapshot_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume_snapshot.VolumeSnapshot()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume_snapshot.VolumeSnapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_volume_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*/snapshots/*}' % client.transport._host, args[1])

def test_get_volume_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_volume_snapshot(volume_snapshot.GetVolumeSnapshotRequest(), name='name_value')

def test_get_volume_snapshot_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume_snapshot.ListVolumeSnapshotsRequest, dict])
def test_list_volume_snapshots_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume_snapshot.ListVolumeSnapshotsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = volume_snapshot.ListVolumeSnapshotsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_volume_snapshots(request)
    assert isinstance(response, pagers.ListVolumeSnapshotsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_volume_snapshots_rest_required_fields(request_type=volume_snapshot.ListVolumeSnapshotsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_volume_snapshots._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_volume_snapshots._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume_snapshot.ListVolumeSnapshotsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume_snapshot.ListVolumeSnapshotsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_volume_snapshots(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_volume_snapshots_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_volume_snapshots._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_volume_snapshots_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_volume_snapshots') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_volume_snapshots') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume_snapshot.ListVolumeSnapshotsRequest.pb(volume_snapshot.ListVolumeSnapshotsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume_snapshot.ListVolumeSnapshotsResponse.to_json(volume_snapshot.ListVolumeSnapshotsResponse())
        request = volume_snapshot.ListVolumeSnapshotsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume_snapshot.ListVolumeSnapshotsResponse()
        client.list_volume_snapshots(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_volume_snapshots_rest_bad_request(transport: str='rest', request_type=volume_snapshot.ListVolumeSnapshotsRequest):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_volume_snapshots(request)

def test_list_volume_snapshots_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume_snapshot.ListVolumeSnapshotsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume_snapshot.ListVolumeSnapshotsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_volume_snapshots(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/volumes/*}/snapshots' % client.transport._host, args[1])

def test_list_volume_snapshots_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_volume_snapshots(volume_snapshot.ListVolumeSnapshotsRequest(), parent='parent_value')

def test_list_volume_snapshots_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()], next_page_token='abc'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[], next_page_token='def'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot()], next_page_token='ghi'), volume_snapshot.ListVolumeSnapshotsResponse(volume_snapshots=[volume_snapshot.VolumeSnapshot(), volume_snapshot.VolumeSnapshot()]))
        response = response + response
        response = tuple((volume_snapshot.ListVolumeSnapshotsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        pager = client.list_volume_snapshots(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, volume_snapshot.VolumeSnapshot) for i in results))
        pages = list(client.list_volume_snapshots(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [lun.GetLunRequest, dict])
def test_get_lun_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/luns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = lun.Lun(name='name_value', id='id_value', state=lun.Lun.State.CREATING, size_gb=739, multiprotocol_type=lun.Lun.MultiprotocolType.LINUX, storage_volume='storage_volume_value', shareable=True, boot_lun=True, storage_type=lun.Lun.StorageType.SSD, wwid='wwid_value', instances=['instances_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = lun.Lun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_lun(request)
    assert isinstance(response, lun.Lun)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.state == lun.Lun.State.CREATING
    assert response.size_gb == 739
    assert response.multiprotocol_type == lun.Lun.MultiprotocolType.LINUX
    assert response.storage_volume == 'storage_volume_value'
    assert response.shareable is True
    assert response.boot_lun is True
    assert response.storage_type == lun.Lun.StorageType.SSD
    assert response.wwid == 'wwid_value'
    assert response.instances == ['instances_value']

def test_get_lun_rest_required_fields(request_type=lun.GetLunRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_lun._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_lun._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = lun.Lun()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = lun.Lun.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_lun(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_lun_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_lun._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_lun_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_lun') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_lun') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = lun.GetLunRequest.pb(lun.GetLunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = lun.Lun.to_json(lun.Lun())
        request = lun.GetLunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = lun.Lun()
        client.get_lun(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_lun_rest_bad_request(transport: str='rest', request_type=lun.GetLunRequest):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/luns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_lun(request)

def test_get_lun_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = lun.Lun()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/luns/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = lun.Lun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_lun(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*/luns/*}' % client.transport._host, args[1])

def test_get_lun_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_lun(lun.GetLunRequest(), name='name_value')

def test_get_lun_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [lun.ListLunsRequest, dict])
def test_list_luns_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = lun.ListLunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = lun.ListLunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_luns(request)
    assert isinstance(response, pagers.ListLunsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_luns_rest_required_fields(request_type=lun.ListLunsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_luns._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_luns._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = lun.ListLunsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = lun.ListLunsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_luns(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_luns_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_luns._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_luns_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_luns') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_luns') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = lun.ListLunsRequest.pb(lun.ListLunsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = lun.ListLunsResponse.to_json(lun.ListLunsResponse())
        request = lun.ListLunsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = lun.ListLunsResponse()
        client.list_luns(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_luns_rest_bad_request(transport: str='rest', request_type=lun.ListLunsRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_luns(request)

def test_list_luns_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = lun.ListLunsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = lun.ListLunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_luns(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/volumes/*}/luns' % client.transport._host, args[1])

def test_list_luns_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_luns(lun.ListLunsRequest(), parent='parent_value')

def test_list_luns_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun(), lun.Lun()], next_page_token='abc'), lun.ListLunsResponse(luns=[], next_page_token='def'), lun.ListLunsResponse(luns=[lun.Lun()], next_page_token='ghi'), lun.ListLunsResponse(luns=[lun.Lun(), lun.Lun()]))
        response = response + response
        response = tuple((lun.ListLunsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        pager = client.list_luns(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, lun.Lun) for i in results))
        pages = list(client.list_luns(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [lun.EvictLunRequest, dict])
def test_evict_lun_rest(request_type):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/luns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.evict_lun(request)
    assert response.operation.name == 'operations/spam'

def test_evict_lun_rest_required_fields(request_type=lun.EvictLunRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).evict_lun._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).evict_lun._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.evict_lun(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_evict_lun_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.evict_lun._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_evict_lun_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_evict_lun') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_evict_lun') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = lun.EvictLunRequest.pb(lun.EvictLunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = lun.EvictLunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.evict_lun(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_evict_lun_rest_bad_request(transport: str='rest', request_type=lun.EvictLunRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/luns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.evict_lun(request)

def test_evict_lun_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/luns/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.evict_lun(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/volumes/*/luns/*}:evict' % client.transport._host, args[1])

def test_evict_lun_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.evict_lun(lun.EvictLunRequest(), name='name_value')

def test_evict_lun_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [nfs_share.GetNfsShareRequest, dict])
def test_get_nfs_share_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = nfs_share.NfsShare(name='name_value', nfs_share_id='nfs_share_id_value', id='id_value', state=nfs_share.NfsShare.State.PROVISIONED, volume='volume_value', requested_size_gib=1917, storage_type=nfs_share.NfsShare.StorageType.SSD)
        response_value = Response()
        response_value.status_code = 200
        return_value = nfs_share.NfsShare.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_nfs_share(request)
    assert isinstance(response, nfs_share.NfsShare)
    assert response.name == 'name_value'
    assert response.nfs_share_id == 'nfs_share_id_value'
    assert response.id == 'id_value'
    assert response.state == nfs_share.NfsShare.State.PROVISIONED
    assert response.volume == 'volume_value'
    assert response.requested_size_gib == 1917
    assert response.storage_type == nfs_share.NfsShare.StorageType.SSD

def test_get_nfs_share_rest_required_fields(request_type=nfs_share.GetNfsShareRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = nfs_share.NfsShare()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = nfs_share.NfsShare.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_nfs_share(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_nfs_share_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_nfs_share._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_nfs_share_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_nfs_share') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_nfs_share') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = nfs_share.GetNfsShareRequest.pb(nfs_share.GetNfsShareRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = nfs_share.NfsShare.to_json(nfs_share.NfsShare())
        request = nfs_share.GetNfsShareRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = nfs_share.NfsShare()
        client.get_nfs_share(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_nfs_share_rest_bad_request(transport: str='rest', request_type=nfs_share.GetNfsShareRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_nfs_share(request)

def test_get_nfs_share_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = nfs_share.NfsShare()
        sample_request = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = nfs_share.NfsShare.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_nfs_share(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/nfsShares/*}' % client.transport._host, args[1])

def test_get_nfs_share_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_nfs_share(nfs_share.GetNfsShareRequest(), name='name_value')

def test_get_nfs_share_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [nfs_share.ListNfsSharesRequest, dict])
def test_list_nfs_shares_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = nfs_share.ListNfsSharesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = nfs_share.ListNfsSharesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_nfs_shares(request)
    assert isinstance(response, pagers.ListNfsSharesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_nfs_shares_rest_required_fields(request_type=nfs_share.ListNfsSharesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_nfs_shares._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_nfs_shares._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = nfs_share.ListNfsSharesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = nfs_share.ListNfsSharesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_nfs_shares(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_nfs_shares_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_nfs_shares._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_nfs_shares_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_nfs_shares') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_nfs_shares') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = nfs_share.ListNfsSharesRequest.pb(nfs_share.ListNfsSharesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = nfs_share.ListNfsSharesResponse.to_json(nfs_share.ListNfsSharesResponse())
        request = nfs_share.ListNfsSharesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = nfs_share.ListNfsSharesResponse()
        client.list_nfs_shares(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_nfs_shares_rest_bad_request(transport: str='rest', request_type=nfs_share.ListNfsSharesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_nfs_shares(request)

def test_list_nfs_shares_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = nfs_share.ListNfsSharesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = nfs_share.ListNfsSharesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_nfs_shares(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/nfsShares' % client.transport._host, args[1])

def test_list_nfs_shares_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_nfs_shares(nfs_share.ListNfsSharesRequest(), parent='parent_value')

def test_list_nfs_shares_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare(), nfs_share.NfsShare()], next_page_token='abc'), nfs_share.ListNfsSharesResponse(nfs_shares=[], next_page_token='def'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare()], next_page_token='ghi'), nfs_share.ListNfsSharesResponse(nfs_shares=[nfs_share.NfsShare(), nfs_share.NfsShare()]))
        response = response + response
        response = tuple((nfs_share.ListNfsSharesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_nfs_shares(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, nfs_share.NfsShare) for i in results))
        pages = list(client.list_nfs_shares(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcb_nfs_share.UpdateNfsShareRequest, dict])
def test_update_nfs_share_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'nfs_share': {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}}
    request_init['nfs_share'] = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3', 'nfs_share_id': 'nfs_share_id_value', 'id': 'id_value', 'state': 1, 'volume': 'volume_value', 'allowed_clients': [{'network': 'network_value', 'share_ip': 'share_ip_value', 'allowed_clients_cidr': 'allowed_clients_cidr_value', 'mount_permissions': 1, 'allow_dev': True, 'allow_suid': True, 'no_root_squash': True, 'nfs_path': 'nfs_path_value'}], 'labels': {}, 'requested_size_gib': 1917, 'storage_type': 1}
    test_field = gcb_nfs_share.UpdateNfsShareRequest.meta.fields['nfs_share']

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
    for (field, value) in request_init['nfs_share'].items():
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
                for i in range(0, len(request_init['nfs_share'][field])):
                    del request_init['nfs_share'][field][i][subfield]
            else:
                del request_init['nfs_share'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_nfs_share(request)
    assert response.operation.name == 'operations/spam'

def test_update_nfs_share_rest_required_fields(request_type=gcb_nfs_share.UpdateNfsShareRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_nfs_share._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_nfs_share(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_nfs_share_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_nfs_share._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('nfsShare',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_nfs_share_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_update_nfs_share') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_update_nfs_share') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_nfs_share.UpdateNfsShareRequest.pb(gcb_nfs_share.UpdateNfsShareRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_nfs_share.UpdateNfsShareRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_nfs_share(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_nfs_share_rest_bad_request(transport: str='rest', request_type=gcb_nfs_share.UpdateNfsShareRequest):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'nfs_share': {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_nfs_share(request)

def test_update_nfs_share_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'nfs_share': {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}}
        mock_args = dict(nfs_share=gcb_nfs_share.NfsShare(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_nfs_share(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{nfs_share.name=projects/*/locations/*/nfsShares/*}' % client.transport._host, args[1])

def test_update_nfs_share_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_nfs_share(gcb_nfs_share.UpdateNfsShareRequest(), nfs_share=gcb_nfs_share.NfsShare(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_nfs_share_rest_error():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcb_nfs_share.CreateNfsShareRequest, dict])
def test_create_nfs_share_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['nfs_share'] = {'name': 'name_value', 'nfs_share_id': 'nfs_share_id_value', 'id': 'id_value', 'state': 1, 'volume': 'volume_value', 'allowed_clients': [{'network': 'network_value', 'share_ip': 'share_ip_value', 'allowed_clients_cidr': 'allowed_clients_cidr_value', 'mount_permissions': 1, 'allow_dev': True, 'allow_suid': True, 'no_root_squash': True, 'nfs_path': 'nfs_path_value'}], 'labels': {}, 'requested_size_gib': 1917, 'storage_type': 1}
    test_field = gcb_nfs_share.CreateNfsShareRequest.meta.fields['nfs_share']

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
    for (field, value) in request_init['nfs_share'].items():
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
                for i in range(0, len(request_init['nfs_share'][field])):
                    del request_init['nfs_share'][field][i][subfield]
            else:
                del request_init['nfs_share'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_nfs_share(request)
    assert response.operation.name == 'operations/spam'

def test_create_nfs_share_rest_required_fields(request_type=gcb_nfs_share.CreateNfsShareRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_nfs_share(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_nfs_share_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_nfs_share._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'nfsShare'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_nfs_share_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_create_nfs_share') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_create_nfs_share') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcb_nfs_share.CreateNfsShareRequest.pb(gcb_nfs_share.CreateNfsShareRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcb_nfs_share.CreateNfsShareRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_nfs_share(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_nfs_share_rest_bad_request(transport: str='rest', request_type=gcb_nfs_share.CreateNfsShareRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_nfs_share(request)

def test_create_nfs_share_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', nfs_share=gcb_nfs_share.NfsShare(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_nfs_share(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/nfsShares' % client.transport._host, args[1])

def test_create_nfs_share_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_nfs_share(gcb_nfs_share.CreateNfsShareRequest(), parent='parent_value', nfs_share=gcb_nfs_share.NfsShare(name='name_value'))

def test_create_nfs_share_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [nfs_share.RenameNfsShareRequest, dict])
def test_rename_nfs_share_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = nfs_share.NfsShare(name='name_value', nfs_share_id='nfs_share_id_value', id='id_value', state=nfs_share.NfsShare.State.PROVISIONED, volume='volume_value', requested_size_gib=1917, storage_type=nfs_share.NfsShare.StorageType.SSD)
        response_value = Response()
        response_value.status_code = 200
        return_value = nfs_share.NfsShare.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.rename_nfs_share(request)
    assert isinstance(response, nfs_share.NfsShare)
    assert response.name == 'name_value'
    assert response.nfs_share_id == 'nfs_share_id_value'
    assert response.id == 'id_value'
    assert response.state == nfs_share.NfsShare.State.PROVISIONED
    assert response.volume == 'volume_value'
    assert response.requested_size_gib == 1917
    assert response.storage_type == nfs_share.NfsShare.StorageType.SSD

def test_rename_nfs_share_rest_required_fields(request_type=nfs_share.RenameNfsShareRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['new_nfsshare_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['newNfsshareId'] = 'new_nfsshare_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'newNfsshareId' in jsonified_request
    assert jsonified_request['newNfsshareId'] == 'new_nfsshare_id_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = nfs_share.NfsShare()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = nfs_share.NfsShare.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.rename_nfs_share(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_rename_nfs_share_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.rename_nfs_share._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'newNfsshareId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_rename_nfs_share_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_rename_nfs_share') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_rename_nfs_share') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = nfs_share.RenameNfsShareRequest.pb(nfs_share.RenameNfsShareRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = nfs_share.NfsShare.to_json(nfs_share.NfsShare())
        request = nfs_share.RenameNfsShareRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = nfs_share.NfsShare()
        client.rename_nfs_share(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_rename_nfs_share_rest_bad_request(transport: str='rest', request_type=nfs_share.RenameNfsShareRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.rename_nfs_share(request)

def test_rename_nfs_share_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = nfs_share.NfsShare()
        sample_request = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
        mock_args = dict(name='name_value', new_nfsshare_id='new_nfsshare_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = nfs_share.NfsShare.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.rename_nfs_share(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/nfsShares/*}:rename' % client.transport._host, args[1])

def test_rename_nfs_share_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.rename_nfs_share(nfs_share.RenameNfsShareRequest(), name='name_value', new_nfsshare_id='new_nfsshare_id_value')

def test_rename_nfs_share_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [nfs_share.DeleteNfsShareRequest, dict])
def test_delete_nfs_share_rest(request_type):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_nfs_share(request)
    assert response.operation.name == 'operations/spam'

def test_delete_nfs_share_rest_required_fields(request_type=nfs_share.DeleteNfsShareRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_nfs_share._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_nfs_share(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_nfs_share_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_nfs_share._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_nfs_share_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_delete_nfs_share') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_delete_nfs_share') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = nfs_share.DeleteNfsShareRequest.pb(nfs_share.DeleteNfsShareRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = nfs_share.DeleteNfsShareRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_nfs_share(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_nfs_share_rest_bad_request(transport: str='rest', request_type=nfs_share.DeleteNfsShareRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_nfs_share(request)

def test_delete_nfs_share_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/nfsShares/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_nfs_share(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/nfsShares/*}' % client.transport._host, args[1])

def test_delete_nfs_share_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_nfs_share(nfs_share.DeleteNfsShareRequest(), name='name_value')

def test_delete_nfs_share_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [provisioning.ListProvisioningQuotasRequest, dict])
def test_list_provisioning_quotas_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ListProvisioningQuotasResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ListProvisioningQuotasResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_provisioning_quotas(request)
    assert isinstance(response, pagers.ListProvisioningQuotasPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_provisioning_quotas_rest_required_fields(request_type=provisioning.ListProvisioningQuotasRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_provisioning_quotas._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_provisioning_quotas._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = provisioning.ListProvisioningQuotasResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = provisioning.ListProvisioningQuotasResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_provisioning_quotas(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_provisioning_quotas_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_provisioning_quotas._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_provisioning_quotas_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_provisioning_quotas') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_provisioning_quotas') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = provisioning.ListProvisioningQuotasRequest.pb(provisioning.ListProvisioningQuotasRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = provisioning.ListProvisioningQuotasResponse.to_json(provisioning.ListProvisioningQuotasResponse())
        request = provisioning.ListProvisioningQuotasRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = provisioning.ListProvisioningQuotasResponse()
        client.list_provisioning_quotas(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_provisioning_quotas_rest_bad_request(transport: str='rest', request_type=provisioning.ListProvisioningQuotasRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_provisioning_quotas(request)

def test_list_provisioning_quotas_rest_flattened():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ListProvisioningQuotasResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ListProvisioningQuotasResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_provisioning_quotas(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/provisioningQuotas' % client.transport._host, args[1])

def test_list_provisioning_quotas_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_provisioning_quotas(provisioning.ListProvisioningQuotasRequest(), parent='parent_value')

def test_list_provisioning_quotas_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()], next_page_token='abc'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[], next_page_token='def'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota()], next_page_token='ghi'), provisioning.ListProvisioningQuotasResponse(provisioning_quotas=[provisioning.ProvisioningQuota(), provisioning.ProvisioningQuota()]))
        response = response + response
        response = tuple((provisioning.ListProvisioningQuotasResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_provisioning_quotas(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, provisioning.ProvisioningQuota) for i in results))
        pages = list(client.list_provisioning_quotas(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [provisioning.SubmitProvisioningConfigRequest, dict])
def test_submit_provisioning_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.SubmitProvisioningConfigResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.SubmitProvisioningConfigResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.submit_provisioning_config(request)
    assert isinstance(response, provisioning.SubmitProvisioningConfigResponse)

def test_submit_provisioning_config_rest_required_fields(request_type=provisioning.SubmitProvisioningConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_provisioning_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_provisioning_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = provisioning.SubmitProvisioningConfigResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = provisioning.SubmitProvisioningConfigResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.submit_provisioning_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_submit_provisioning_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.submit_provisioning_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'provisioningConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_submit_provisioning_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_submit_provisioning_config') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_submit_provisioning_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = provisioning.SubmitProvisioningConfigRequest.pb(provisioning.SubmitProvisioningConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = provisioning.SubmitProvisioningConfigResponse.to_json(provisioning.SubmitProvisioningConfigResponse())
        request = provisioning.SubmitProvisioningConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = provisioning.SubmitProvisioningConfigResponse()
        client.submit_provisioning_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_submit_provisioning_config_rest_bad_request(transport: str='rest', request_type=provisioning.SubmitProvisioningConfigRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.submit_provisioning_config(request)

def test_submit_provisioning_config_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.SubmitProvisioningConfigResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.SubmitProvisioningConfigResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.submit_provisioning_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/provisioningConfigs:submit' % client.transport._host, args[1])

def test_submit_provisioning_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.submit_provisioning_config(provisioning.SubmitProvisioningConfigRequest(), parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))

def test_submit_provisioning_config_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [provisioning.GetProvisioningConfigRequest, dict])
def test_get_provisioning_config_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ProvisioningConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_provisioning_config(request)
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

def test_get_provisioning_config_rest_required_fields(request_type=provisioning.GetProvisioningConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_provisioning_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_provisioning_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = provisioning.ProvisioningConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = provisioning.ProvisioningConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_provisioning_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_provisioning_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_provisioning_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_provisioning_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_get_provisioning_config') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_get_provisioning_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = provisioning.GetProvisioningConfigRequest.pb(provisioning.GetProvisioningConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = provisioning.ProvisioningConfig.to_json(provisioning.ProvisioningConfig())
        request = provisioning.GetProvisioningConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = provisioning.ProvisioningConfig()
        client.get_provisioning_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_provisioning_config_rest_bad_request(transport: str='rest', request_type=provisioning.GetProvisioningConfigRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_provisioning_config(request)

def test_get_provisioning_config_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ProvisioningConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ProvisioningConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_provisioning_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/provisioningConfigs/*}' % client.transport._host, args[1])

def test_get_provisioning_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_provisioning_config(provisioning.GetProvisioningConfigRequest(), name='name_value')

def test_get_provisioning_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [provisioning.CreateProvisioningConfigRequest, dict])
def test_create_provisioning_config_rest(request_type):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['provisioning_config'] = {'name': 'name_value', 'instances': [{'name': 'name_value', 'id': 'id_value', 'instance_type': 'instance_type_value', 'hyperthreading': True, 'os_image': 'os_image_value', 'client_network': {'network_id': 'network_id_value', 'address': 'address_value', 'existing_network_id': 'existing_network_id_value'}, 'private_network': {}, 'user_note': 'user_note_value', 'account_networks_enabled': True, 'network_config': 1, 'network_template': 'network_template_value', 'logical_interfaces': [{'logical_network_interfaces': [{'network': 'network_value', 'ip_address': 'ip_address_value', 'default_gateway': True, 'network_type': 1, 'id': 'id_value'}], 'name': 'name_value', 'interface_index': 1576}], 'ssh_key_names': ['ssh_key_names_value1', 'ssh_key_names_value2']}], 'networks': [{'name': 'name_value', 'id': 'id_value', 'type_': 1, 'bandwidth': 1, 'vlan_attachments': [{'id': 'id_value', 'pairing_key': 'pairing_key_value'}], 'cidr': 'cidr_value', 'service_cidr': 1, 'user_note': 'user_note_value', 'gcp_service': 'gcp_service_value', 'vlan_same_project': True, 'jumbo_frames_enabled': True}], 'volumes': [{'name': 'name_value', 'id': 'id_value', 'snapshots_enabled': True, 'type_': 1, 'protocol': 1, 'size_gb': 739, 'lun_ranges': [{'quantity': 895, 'size_gb': 739}], 'machine_ids': ['machine_ids_value1', 'machine_ids_value2'], 'nfs_exports': [{'network_id': 'network_id_value', 'machine_id': 'machine_id_value', 'cidr': 'cidr_value', 'permissions': 1, 'no_root_squash': True, 'allow_suid': True, 'allow_dev': True}], 'user_note': 'user_note_value', 'gcp_service': 'gcp_service_value', 'performance_tier': 1}], 'ticket_id': 'ticket_id_value', 'handover_service_account': 'handover_service_account_value', 'email': 'email_value', 'state': 1, 'location': 'location_value', 'update_time': {'seconds': 751, 'nanos': 543}, 'cloud_console_uri': 'cloud_console_uri_value', 'vpc_sc_enabled': True, 'status_message': 'status_message_value', 'custom_id': 'custom_id_value'}
    test_field = provisioning.CreateProvisioningConfigRequest.meta.fields['provisioning_config']

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
    for (field, value) in request_init['provisioning_config'].items():
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
                for i in range(0, len(request_init['provisioning_config'][field])):
                    del request_init['provisioning_config'][field][i][subfield]
            else:
                del request_init['provisioning_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ProvisioningConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_provisioning_config(request)
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

def test_create_provisioning_config_rest_required_fields(request_type=provisioning.CreateProvisioningConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_provisioning_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_provisioning_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('email',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = provisioning.ProvisioningConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = provisioning.ProvisioningConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_provisioning_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_provisioning_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_provisioning_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('email',)) & set(('parent', 'provisioningConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_provisioning_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_create_provisioning_config') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_create_provisioning_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = provisioning.CreateProvisioningConfigRequest.pb(provisioning.CreateProvisioningConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = provisioning.ProvisioningConfig.to_json(provisioning.ProvisioningConfig())
        request = provisioning.CreateProvisioningConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = provisioning.ProvisioningConfig()
        client.create_provisioning_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_provisioning_config_rest_bad_request(transport: str='rest', request_type=provisioning.CreateProvisioningConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_provisioning_config(request)

def test_create_provisioning_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ProvisioningConfig()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ProvisioningConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_provisioning_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/provisioningConfigs' % client.transport._host, args[1])

def test_create_provisioning_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_provisioning_config(provisioning.CreateProvisioningConfigRequest(), parent='parent_value', provisioning_config=provisioning.ProvisioningConfig(name='name_value'))

def test_create_provisioning_config_rest_error():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [provisioning.UpdateProvisioningConfigRequest, dict])
def test_update_provisioning_config_rest(request_type):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'provisioning_config': {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3'}}
    request_init['provisioning_config'] = {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3', 'instances': [{'name': 'name_value', 'id': 'id_value', 'instance_type': 'instance_type_value', 'hyperthreading': True, 'os_image': 'os_image_value', 'client_network': {'network_id': 'network_id_value', 'address': 'address_value', 'existing_network_id': 'existing_network_id_value'}, 'private_network': {}, 'user_note': 'user_note_value', 'account_networks_enabled': True, 'network_config': 1, 'network_template': 'network_template_value', 'logical_interfaces': [{'logical_network_interfaces': [{'network': 'network_value', 'ip_address': 'ip_address_value', 'default_gateway': True, 'network_type': 1, 'id': 'id_value'}], 'name': 'name_value', 'interface_index': 1576}], 'ssh_key_names': ['ssh_key_names_value1', 'ssh_key_names_value2']}], 'networks': [{'name': 'name_value', 'id': 'id_value', 'type_': 1, 'bandwidth': 1, 'vlan_attachments': [{'id': 'id_value', 'pairing_key': 'pairing_key_value'}], 'cidr': 'cidr_value', 'service_cidr': 1, 'user_note': 'user_note_value', 'gcp_service': 'gcp_service_value', 'vlan_same_project': True, 'jumbo_frames_enabled': True}], 'volumes': [{'name': 'name_value', 'id': 'id_value', 'snapshots_enabled': True, 'type_': 1, 'protocol': 1, 'size_gb': 739, 'lun_ranges': [{'quantity': 895, 'size_gb': 739}], 'machine_ids': ['machine_ids_value1', 'machine_ids_value2'], 'nfs_exports': [{'network_id': 'network_id_value', 'machine_id': 'machine_id_value', 'cidr': 'cidr_value', 'permissions': 1, 'no_root_squash': True, 'allow_suid': True, 'allow_dev': True}], 'user_note': 'user_note_value', 'gcp_service': 'gcp_service_value', 'performance_tier': 1}], 'ticket_id': 'ticket_id_value', 'handover_service_account': 'handover_service_account_value', 'email': 'email_value', 'state': 1, 'location': 'location_value', 'update_time': {'seconds': 751, 'nanos': 543}, 'cloud_console_uri': 'cloud_console_uri_value', 'vpc_sc_enabled': True, 'status_message': 'status_message_value', 'custom_id': 'custom_id_value'}
    test_field = provisioning.UpdateProvisioningConfigRequest.meta.fields['provisioning_config']

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
    for (field, value) in request_init['provisioning_config'].items():
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
                for i in range(0, len(request_init['provisioning_config'][field])):
                    del request_init['provisioning_config'][field][i][subfield]
            else:
                del request_init['provisioning_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ProvisioningConfig(name='name_value', ticket_id='ticket_id_value', handover_service_account='handover_service_account_value', email='email_value', state=provisioning.ProvisioningConfig.State.DRAFT, location='location_value', cloud_console_uri='cloud_console_uri_value', vpc_sc_enabled=True, status_message='status_message_value', custom_id='custom_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ProvisioningConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_provisioning_config(request)
    assert isinstance(response, provisioning.ProvisioningConfig)
    assert response.name == 'name_value'
    assert response.ticket_id == 'ticket_id_value'
    assert response.handover_service_account == 'handover_service_account_value'
    assert response.email == 'email_value'
    assert response.state == provisioning.ProvisioningConfig.State.DRAFT
    assert response.location == 'location_value'
    assert response.cloud_console_uri == 'cloud_console_uri_value'
    assert response.vpc_sc_enabled is True
    assert response.status_message == 'status_message_value'
    assert response.custom_id == 'custom_id_value'

def test_update_provisioning_config_rest_required_fields(request_type=provisioning.UpdateProvisioningConfigRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_provisioning_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_provisioning_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('email', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = provisioning.ProvisioningConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = provisioning.ProvisioningConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_provisioning_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_provisioning_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_provisioning_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('email', 'updateMask')) & set(('provisioningConfig', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_provisioning_config_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_update_provisioning_config') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_update_provisioning_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = provisioning.UpdateProvisioningConfigRequest.pb(provisioning.UpdateProvisioningConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = provisioning.ProvisioningConfig.to_json(provisioning.ProvisioningConfig())
        request = provisioning.UpdateProvisioningConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = provisioning.ProvisioningConfig()
        client.update_provisioning_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_provisioning_config_rest_bad_request(transport: str='rest', request_type=provisioning.UpdateProvisioningConfigRequest):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'provisioning_config': {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_provisioning_config(request)

def test_update_provisioning_config_rest_flattened():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = provisioning.ProvisioningConfig()
        sample_request = {'provisioning_config': {'name': 'projects/sample1/locations/sample2/provisioningConfigs/sample3'}}
        mock_args = dict(provisioning_config=provisioning.ProvisioningConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = provisioning.ProvisioningConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_provisioning_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{provisioning_config.name=projects/*/locations/*/provisioningConfigs/*}' % client.transport._host, args[1])

def test_update_provisioning_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_provisioning_config(provisioning.UpdateProvisioningConfigRequest(), provisioning_config=provisioning.ProvisioningConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_provisioning_config_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [network.RenameNetworkRequest, dict])
def test_rename_network_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/networks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.Network(name='name_value', id='id_value', type_=network.Network.Type.CLIENT, ip_address='ip_address_value', mac_address=['mac_address_value'], state=network.Network.State.PROVISIONING, vlan_id='vlan_id_value', cidr='cidr_value', services_cidr='services_cidr_value', pod='pod_value', jumbo_frames_enabled=True, gateway_ip='gateway_ip_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = network.Network.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.rename_network(request)
    assert isinstance(response, network.Network)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.type_ == network.Network.Type.CLIENT
    assert response.ip_address == 'ip_address_value'
    assert response.mac_address == ['mac_address_value']
    assert response.state == network.Network.State.PROVISIONING
    assert response.vlan_id == 'vlan_id_value'
    assert response.cidr == 'cidr_value'
    assert response.services_cidr == 'services_cidr_value'
    assert response.pod == 'pod_value'
    assert response.jumbo_frames_enabled is True
    assert response.gateway_ip == 'gateway_ip_value'

def test_rename_network_rest_required_fields(request_type=network.RenameNetworkRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['new_network_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['newNetworkId'] = 'new_network_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rename_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'newNetworkId' in jsonified_request
    assert jsonified_request['newNetworkId'] == 'new_network_id_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = network.Network()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = network.Network.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.rename_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_rename_network_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.rename_network._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'newNetworkId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_rename_network_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_rename_network') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_rename_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = network.RenameNetworkRequest.pb(network.RenameNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = network.Network.to_json(network.Network())
        request = network.RenameNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = network.Network()
        client.rename_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_rename_network_rest_bad_request(transport: str='rest', request_type=network.RenameNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/networks/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.rename_network(request)

def test_rename_network_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = network.Network()
        sample_request = {'name': 'projects/sample1/locations/sample2/networks/sample3'}
        mock_args = dict(name='name_value', new_network_id='new_network_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = network.Network.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.rename_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/networks/*}:rename' % client.transport._host, args[1])

def test_rename_network_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.rename_network(network.RenameNetworkRequest(), name='name_value', new_network_id='new_network_id_value')

def test_rename_network_rest_error():
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [osimage.ListOSImagesRequest, dict])
def test_list_os_images_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = osimage.ListOSImagesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = osimage.ListOSImagesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_os_images(request)
    assert isinstance(response, pagers.ListOSImagesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_images_rest_required_fields(request_type=osimage.ListOSImagesRequest):
    if False:
        return 10
    transport_class = transports.BareMetalSolutionRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_images._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_images._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = osimage.ListOSImagesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = osimage.ListOSImagesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_os_images(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_os_images_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_os_images._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_os_images_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BareMetalSolutionRestInterceptor())
    client = BareMetalSolutionClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'post_list_os_images') as post, mock.patch.object(transports.BareMetalSolutionRestInterceptor, 'pre_list_os_images') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = osimage.ListOSImagesRequest.pb(osimage.ListOSImagesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = osimage.ListOSImagesResponse.to_json(osimage.ListOSImagesResponse())
        request = osimage.ListOSImagesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = osimage.ListOSImagesResponse()
        client.list_os_images(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_os_images_rest_bad_request(transport: str='rest', request_type=osimage.ListOSImagesRequest):
    if False:
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_os_images(request)

def test_list_os_images_rest_flattened():
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = osimage.ListOSImagesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = osimage.ListOSImagesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_os_images(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/osImages' % client.transport._host, args[1])

def test_list_os_images_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_os_images(osimage.ListOSImagesRequest(), parent='parent_value')

def test_list_os_images_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage(), osimage.OSImage()], next_page_token='abc'), osimage.ListOSImagesResponse(os_images=[], next_page_token='def'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage()], next_page_token='ghi'), osimage.ListOSImagesResponse(os_images=[osimage.OSImage(), osimage.OSImage()]))
        response = response + response
        response = tuple((osimage.ListOSImagesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_os_images(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, osimage.OSImage) for i in results))
        pages = list(client.list_os_images(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BareMetalSolutionGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.BareMetalSolutionGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = BareMetalSolutionClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.BareMetalSolutionGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = BareMetalSolutionClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = BareMetalSolutionClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.BareMetalSolutionGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = BareMetalSolutionClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.BareMetalSolutionGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = BareMetalSolutionClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.BareMetalSolutionGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.BareMetalSolutionGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionGrpcAsyncIOTransport, transports.BareMetalSolutionRestTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        return 10
    transport = BareMetalSolutionClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.BareMetalSolutionGrpcTransport)

def test_bare_metal_solution_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.BareMetalSolutionTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_bare_metal_solution_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.bare_metal_solution_v2.services.bare_metal_solution.transports.BareMetalSolutionTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.BareMetalSolutionTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_instances', 'get_instance', 'update_instance', 'rename_instance', 'reset_instance', 'start_instance', 'stop_instance', 'enable_interactive_serial_console', 'disable_interactive_serial_console', 'detach_lun', 'list_ssh_keys', 'create_ssh_key', 'delete_ssh_key', 'list_volumes', 'get_volume', 'update_volume', 'rename_volume', 'evict_volume', 'resize_volume', 'list_networks', 'list_network_usage', 'get_network', 'update_network', 'create_volume_snapshot', 'restore_volume_snapshot', 'delete_volume_snapshot', 'get_volume_snapshot', 'list_volume_snapshots', 'get_lun', 'list_luns', 'evict_lun', 'get_nfs_share', 'list_nfs_shares', 'update_nfs_share', 'create_nfs_share', 'rename_nfs_share', 'delete_nfs_share', 'list_provisioning_quotas', 'submit_provisioning_config', 'get_provisioning_config', 'create_provisioning_config', 'update_provisioning_config', 'rename_network', 'list_os_images', 'get_location', 'list_locations')
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

def test_bare_metal_solution_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.bare_metal_solution_v2.services.bare_metal_solution.transports.BareMetalSolutionTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.BareMetalSolutionTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_bare_metal_solution_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.bare_metal_solution_v2.services.bare_metal_solution.transports.BareMetalSolutionTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.BareMetalSolutionTransport()
        adc.assert_called_once()

def test_bare_metal_solution_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        BareMetalSolutionClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionGrpcAsyncIOTransport])
def test_bare_metal_solution_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionGrpcAsyncIOTransport, transports.BareMetalSolutionRestTransport])
def test_bare_metal_solution_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.BareMetalSolutionGrpcTransport, grpc_helpers), (transports.BareMetalSolutionGrpcAsyncIOTransport, grpc_helpers_async)])
def test_bare_metal_solution_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('baremetalsolution.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='baremetalsolution.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionGrpcAsyncIOTransport])
def test_bare_metal_solution_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        i = 10
        return i + 15
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

def test_bare_metal_solution_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.BareMetalSolutionRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_bare_metal_solution_rest_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_bare_metal_solution_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='baremetalsolution.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('baremetalsolution.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://baremetalsolution.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_bare_metal_solution_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='baremetalsolution.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('baremetalsolution.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://baremetalsolution.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_bare_metal_solution_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = BareMetalSolutionClient(credentials=creds1, transport=transport_name)
    client2 = BareMetalSolutionClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_instances._session
    session2 = client2.transport.list_instances._session
    assert session1 != session2
    session1 = client1.transport.get_instance._session
    session2 = client2.transport.get_instance._session
    assert session1 != session2
    session1 = client1.transport.update_instance._session
    session2 = client2.transport.update_instance._session
    assert session1 != session2
    session1 = client1.transport.rename_instance._session
    session2 = client2.transport.rename_instance._session
    assert session1 != session2
    session1 = client1.transport.reset_instance._session
    session2 = client2.transport.reset_instance._session
    assert session1 != session2
    session1 = client1.transport.start_instance._session
    session2 = client2.transport.start_instance._session
    assert session1 != session2
    session1 = client1.transport.stop_instance._session
    session2 = client2.transport.stop_instance._session
    assert session1 != session2
    session1 = client1.transport.enable_interactive_serial_console._session
    session2 = client2.transport.enable_interactive_serial_console._session
    assert session1 != session2
    session1 = client1.transport.disable_interactive_serial_console._session
    session2 = client2.transport.disable_interactive_serial_console._session
    assert session1 != session2
    session1 = client1.transport.detach_lun._session
    session2 = client2.transport.detach_lun._session
    assert session1 != session2
    session1 = client1.transport.list_ssh_keys._session
    session2 = client2.transport.list_ssh_keys._session
    assert session1 != session2
    session1 = client1.transport.create_ssh_key._session
    session2 = client2.transport.create_ssh_key._session
    assert session1 != session2
    session1 = client1.transport.delete_ssh_key._session
    session2 = client2.transport.delete_ssh_key._session
    assert session1 != session2
    session1 = client1.transport.list_volumes._session
    session2 = client2.transport.list_volumes._session
    assert session1 != session2
    session1 = client1.transport.get_volume._session
    session2 = client2.transport.get_volume._session
    assert session1 != session2
    session1 = client1.transport.update_volume._session
    session2 = client2.transport.update_volume._session
    assert session1 != session2
    session1 = client1.transport.rename_volume._session
    session2 = client2.transport.rename_volume._session
    assert session1 != session2
    session1 = client1.transport.evict_volume._session
    session2 = client2.transport.evict_volume._session
    assert session1 != session2
    session1 = client1.transport.resize_volume._session
    session2 = client2.transport.resize_volume._session
    assert session1 != session2
    session1 = client1.transport.list_networks._session
    session2 = client2.transport.list_networks._session
    assert session1 != session2
    session1 = client1.transport.list_network_usage._session
    session2 = client2.transport.list_network_usage._session
    assert session1 != session2
    session1 = client1.transport.get_network._session
    session2 = client2.transport.get_network._session
    assert session1 != session2
    session1 = client1.transport.update_network._session
    session2 = client2.transport.update_network._session
    assert session1 != session2
    session1 = client1.transport.create_volume_snapshot._session
    session2 = client2.transport.create_volume_snapshot._session
    assert session1 != session2
    session1 = client1.transport.restore_volume_snapshot._session
    session2 = client2.transport.restore_volume_snapshot._session
    assert session1 != session2
    session1 = client1.transport.delete_volume_snapshot._session
    session2 = client2.transport.delete_volume_snapshot._session
    assert session1 != session2
    session1 = client1.transport.get_volume_snapshot._session
    session2 = client2.transport.get_volume_snapshot._session
    assert session1 != session2
    session1 = client1.transport.list_volume_snapshots._session
    session2 = client2.transport.list_volume_snapshots._session
    assert session1 != session2
    session1 = client1.transport.get_lun._session
    session2 = client2.transport.get_lun._session
    assert session1 != session2
    session1 = client1.transport.list_luns._session
    session2 = client2.transport.list_luns._session
    assert session1 != session2
    session1 = client1.transport.evict_lun._session
    session2 = client2.transport.evict_lun._session
    assert session1 != session2
    session1 = client1.transport.get_nfs_share._session
    session2 = client2.transport.get_nfs_share._session
    assert session1 != session2
    session1 = client1.transport.list_nfs_shares._session
    session2 = client2.transport.list_nfs_shares._session
    assert session1 != session2
    session1 = client1.transport.update_nfs_share._session
    session2 = client2.transport.update_nfs_share._session
    assert session1 != session2
    session1 = client1.transport.create_nfs_share._session
    session2 = client2.transport.create_nfs_share._session
    assert session1 != session2
    session1 = client1.transport.rename_nfs_share._session
    session2 = client2.transport.rename_nfs_share._session
    assert session1 != session2
    session1 = client1.transport.delete_nfs_share._session
    session2 = client2.transport.delete_nfs_share._session
    assert session1 != session2
    session1 = client1.transport.list_provisioning_quotas._session
    session2 = client2.transport.list_provisioning_quotas._session
    assert session1 != session2
    session1 = client1.transport.submit_provisioning_config._session
    session2 = client2.transport.submit_provisioning_config._session
    assert session1 != session2
    session1 = client1.transport.get_provisioning_config._session
    session2 = client2.transport.get_provisioning_config._session
    assert session1 != session2
    session1 = client1.transport.create_provisioning_config._session
    session2 = client2.transport.create_provisioning_config._session
    assert session1 != session2
    session1 = client1.transport.update_provisioning_config._session
    session2 = client2.transport.update_provisioning_config._session
    assert session1 != session2
    session1 = client1.transport.rename_network._session
    session2 = client2.transport.rename_network._session
    assert session1 != session2
    session1 = client1.transport.list_os_images._session
    session2 = client2.transport.list_os_images._session
    assert session1 != session2

def test_bare_metal_solution_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.BareMetalSolutionGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_bare_metal_solution_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.BareMetalSolutionGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionGrpcAsyncIOTransport])
def test_bare_metal_solution_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.BareMetalSolutionGrpcTransport, transports.BareMetalSolutionGrpcAsyncIOTransport])
def test_bare_metal_solution_transport_channel_mtls_with_adc(transport_class):
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

def test_bare_metal_solution_grpc_lro_client():
    if False:
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_bare_metal_solution_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_instance_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    instance = 'whelk'
    expected = 'projects/{project}/locations/{location}/instances/{instance}'.format(project=project, location=location, instance=instance)
    actual = BareMetalSolutionClient.instance_path(project, location, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'instance': 'nudibranch'}
    path = BareMetalSolutionClient.instance_path(**expected)
    actual = BareMetalSolutionClient.parse_instance_path(path)
    assert expected == actual

def test_instance_config_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    instance_config = 'winkle'
    expected = 'projects/{project}/locations/{location}/instanceConfigs/{instance_config}'.format(project=project, location=location, instance_config=instance_config)
    actual = BareMetalSolutionClient.instance_config_path(project, location, instance_config)
    assert expected == actual

def test_parse_instance_config_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'instance_config': 'abalone'}
    path = BareMetalSolutionClient.instance_config_path(**expected)
    actual = BareMetalSolutionClient.parse_instance_config_path(path)
    assert expected == actual

def test_instance_quota_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    instance_quota = 'whelk'
    expected = 'projects/{project}/locations/{location}/instanceQuotas/{instance_quota}'.format(project=project, location=location, instance_quota=instance_quota)
    actual = BareMetalSolutionClient.instance_quota_path(project, location, instance_quota)
    assert expected == actual

def test_parse_instance_quota_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'instance_quota': 'nudibranch'}
    path = BareMetalSolutionClient.instance_quota_path(**expected)
    actual = BareMetalSolutionClient.parse_instance_quota_path(path)
    assert expected == actual

def test_interconnect_attachment_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    region = 'mussel'
    interconnect_attachment = 'winkle'
    expected = 'projects/{project}/regions/{region}/interconnectAttachments/{interconnect_attachment}'.format(project=project, region=region, interconnect_attachment=interconnect_attachment)
    actual = BareMetalSolutionClient.interconnect_attachment_path(project, region, interconnect_attachment)
    assert expected == actual

def test_parse_interconnect_attachment_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'region': 'scallop', 'interconnect_attachment': 'abalone'}
    path = BareMetalSolutionClient.interconnect_attachment_path(**expected)
    actual = BareMetalSolutionClient.parse_interconnect_attachment_path(path)
    assert expected == actual

def test_lun_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    volume = 'whelk'
    lun = 'octopus'
    expected = 'projects/{project}/locations/{location}/volumes/{volume}/luns/{lun}'.format(project=project, location=location, volume=volume, lun=lun)
    actual = BareMetalSolutionClient.lun_path(project, location, volume, lun)
    assert expected == actual

def test_parse_lun_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'volume': 'cuttlefish', 'lun': 'mussel'}
    path = BareMetalSolutionClient.lun_path(**expected)
    actual = BareMetalSolutionClient.parse_lun_path(path)
    assert expected == actual

def test_network_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    network = 'scallop'
    expected = 'projects/{project}/locations/{location}/networks/{network}'.format(project=project, location=location, network=network)
    actual = BareMetalSolutionClient.network_path(project, location, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'network': 'clam'}
    path = BareMetalSolutionClient.network_path(**expected)
    actual = BareMetalSolutionClient.parse_network_path(path)
    assert expected == actual

def test_network_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    network_config = 'oyster'
    expected = 'projects/{project}/locations/{location}/networkConfigs/{network_config}'.format(project=project, location=location, network_config=network_config)
    actual = BareMetalSolutionClient.network_config_path(project, location, network_config)
    assert expected == actual

def test_parse_network_config_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'network_config': 'mussel'}
    path = BareMetalSolutionClient.network_config_path(**expected)
    actual = BareMetalSolutionClient.parse_network_config_path(path)
    assert expected == actual

def test_nfs_share_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    nfs_share = 'scallop'
    expected = 'projects/{project}/locations/{location}/nfsShares/{nfs_share}'.format(project=project, location=location, nfs_share=nfs_share)
    actual = BareMetalSolutionClient.nfs_share_path(project, location, nfs_share)
    assert expected == actual

def test_parse_nfs_share_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'nfs_share': 'clam'}
    path = BareMetalSolutionClient.nfs_share_path(**expected)
    actual = BareMetalSolutionClient.parse_nfs_share_path(path)
    assert expected == actual

def test_os_image_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    os_image = 'oyster'
    expected = 'projects/{project}/locations/{location}/osImages/{os_image}'.format(project=project, location=location, os_image=os_image)
    actual = BareMetalSolutionClient.os_image_path(project, location, os_image)
    assert expected == actual

def test_parse_os_image_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'os_image': 'mussel'}
    path = BareMetalSolutionClient.os_image_path(**expected)
    actual = BareMetalSolutionClient.parse_os_image_path(path)
    assert expected == actual

def test_provisioning_config_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    provisioning_config = 'scallop'
    expected = 'projects/{project}/locations/{location}/provisioningConfigs/{provisioning_config}'.format(project=project, location=location, provisioning_config=provisioning_config)
    actual = BareMetalSolutionClient.provisioning_config_path(project, location, provisioning_config)
    assert expected == actual

def test_parse_provisioning_config_path():
    if False:
        return 10
    expected = {'project': 'abalone', 'location': 'squid', 'provisioning_config': 'clam'}
    path = BareMetalSolutionClient.provisioning_config_path(**expected)
    actual = BareMetalSolutionClient.parse_provisioning_config_path(path)
    assert expected == actual

def test_provisioning_quota_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    provisioning_quota = 'oyster'
    expected = 'projects/{project}/locations/{location}/provisioningQuotas/{provisioning_quota}'.format(project=project, location=location, provisioning_quota=provisioning_quota)
    actual = BareMetalSolutionClient.provisioning_quota_path(project, location, provisioning_quota)
    assert expected == actual

def test_parse_provisioning_quota_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'provisioning_quota': 'mussel'}
    path = BareMetalSolutionClient.provisioning_quota_path(**expected)
    actual = BareMetalSolutionClient.parse_provisioning_quota_path(path)
    assert expected == actual

def test_server_network_template_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    server_network_template = 'scallop'
    expected = 'projects/{project}/locations/{location}/serverNetworkTemplate/{server_network_template}'.format(project=project, location=location, server_network_template=server_network_template)
    actual = BareMetalSolutionClient.server_network_template_path(project, location, server_network_template)
    assert expected == actual

def test_parse_server_network_template_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone', 'location': 'squid', 'server_network_template': 'clam'}
    path = BareMetalSolutionClient.server_network_template_path(**expected)
    actual = BareMetalSolutionClient.parse_server_network_template_path(path)
    assert expected == actual

def test_ssh_key_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    ssh_key = 'oyster'
    expected = 'projects/{project}/locations/{location}/sshKeys/{ssh_key}'.format(project=project, location=location, ssh_key=ssh_key)
    actual = BareMetalSolutionClient.ssh_key_path(project, location, ssh_key)
    assert expected == actual

def test_parse_ssh_key_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'ssh_key': 'mussel'}
    path = BareMetalSolutionClient.ssh_key_path(**expected)
    actual = BareMetalSolutionClient.parse_ssh_key_path(path)
    assert expected == actual

def test_volume_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    volume = 'scallop'
    expected = 'projects/{project}/locations/{location}/volumes/{volume}'.format(project=project, location=location, volume=volume)
    actual = BareMetalSolutionClient.volume_path(project, location, volume)
    assert expected == actual

def test_parse_volume_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'volume': 'clam'}
    path = BareMetalSolutionClient.volume_path(**expected)
    actual = BareMetalSolutionClient.parse_volume_path(path)
    assert expected == actual

def test_volume_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    volume_config = 'oyster'
    expected = 'projects/{project}/locations/{location}/volumeConfigs/{volume_config}'.format(project=project, location=location, volume_config=volume_config)
    actual = BareMetalSolutionClient.volume_config_path(project, location, volume_config)
    assert expected == actual

def test_parse_volume_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'volume_config': 'mussel'}
    path = BareMetalSolutionClient.volume_config_path(**expected)
    actual = BareMetalSolutionClient.parse_volume_config_path(path)
    assert expected == actual

def test_volume_snapshot_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    volume = 'scallop'
    snapshot = 'abalone'
    expected = 'projects/{project}/locations/{location}/volumes/{volume}/snapshots/{snapshot}'.format(project=project, location=location, volume=volume, snapshot=snapshot)
    actual = BareMetalSolutionClient.volume_snapshot_path(project, location, volume, snapshot)
    assert expected == actual

def test_parse_volume_snapshot_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam', 'volume': 'whelk', 'snapshot': 'octopus'}
    path = BareMetalSolutionClient.volume_snapshot_path(**expected)
    actual = BareMetalSolutionClient.parse_volume_snapshot_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = BareMetalSolutionClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nudibranch'}
    path = BareMetalSolutionClient.common_billing_account_path(**expected)
    actual = BareMetalSolutionClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = BareMetalSolutionClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'mussel'}
    path = BareMetalSolutionClient.common_folder_path(**expected)
    actual = BareMetalSolutionClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = BareMetalSolutionClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nautilus'}
    path = BareMetalSolutionClient.common_organization_path(**expected)
    actual = BareMetalSolutionClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = BareMetalSolutionClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'abalone'}
    path = BareMetalSolutionClient.common_project_path(**expected)
    actual = BareMetalSolutionClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = BareMetalSolutionClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = BareMetalSolutionClient.common_location_path(**expected)
    actual = BareMetalSolutionClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.BareMetalSolutionTransport, '_prep_wrapped_messages') as prep:
        client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.BareMetalSolutionTransport, '_prep_wrapped_messages') as prep:
        transport_class = BareMetalSolutionClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = BareMetalSolutionAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = BareMetalSolutionClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(BareMetalSolutionClient, transports.BareMetalSolutionGrpcTransport), (BareMetalSolutionAsyncClient, transports.BareMetalSolutionGrpcAsyncIOTransport)])
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