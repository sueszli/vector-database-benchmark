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
from google.cloud.dataproc_v1.services.node_group_controller import NodeGroupControllerAsyncClient, NodeGroupControllerClient, transports
from google.cloud.dataproc_v1.types import clusters, node_groups, operations

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert NodeGroupControllerClient._get_default_mtls_endpoint(None) is None
    assert NodeGroupControllerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert NodeGroupControllerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert NodeGroupControllerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert NodeGroupControllerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert NodeGroupControllerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(NodeGroupControllerClient, 'grpc'), (NodeGroupControllerAsyncClient, 'grpc_asyncio'), (NodeGroupControllerClient, 'rest')])
def test_node_group_controller_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.NodeGroupControllerGrpcTransport, 'grpc'), (transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.NodeGroupControllerRestTransport, 'rest')])
def test_node_group_controller_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(NodeGroupControllerClient, 'grpc'), (NodeGroupControllerAsyncClient, 'grpc_asyncio'), (NodeGroupControllerClient, 'rest')])
def test_node_group_controller_client_from_service_account_file(client_class, transport_name):
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

def test_node_group_controller_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = NodeGroupControllerClient.get_transport_class()
    available_transports = [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerRestTransport]
    assert transport in available_transports
    transport = NodeGroupControllerClient.get_transport_class('grpc')
    assert transport == transports.NodeGroupControllerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport, 'grpc'), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (NodeGroupControllerClient, transports.NodeGroupControllerRestTransport, 'rest')])
@mock.patch.object(NodeGroupControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupControllerClient))
@mock.patch.object(NodeGroupControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupControllerAsyncClient))
def test_node_group_controller_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(NodeGroupControllerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(NodeGroupControllerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport, 'grpc', 'true'), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport, 'grpc', 'false'), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (NodeGroupControllerClient, transports.NodeGroupControllerRestTransport, 'rest', 'true'), (NodeGroupControllerClient, transports.NodeGroupControllerRestTransport, 'rest', 'false')])
@mock.patch.object(NodeGroupControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupControllerClient))
@mock.patch.object(NodeGroupControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupControllerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_node_group_controller_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class', [NodeGroupControllerClient, NodeGroupControllerAsyncClient])
@mock.patch.object(NodeGroupControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupControllerClient))
@mock.patch.object(NodeGroupControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupControllerAsyncClient))
def test_node_group_controller_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        return 10
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport, 'grpc'), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (NodeGroupControllerClient, transports.NodeGroupControllerRestTransport, 'rest')])
def test_node_group_controller_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport, 'grpc', grpc_helpers), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (NodeGroupControllerClient, transports.NodeGroupControllerRestTransport, 'rest', None)])
def test_node_group_controller_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_node_group_controller_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dataproc_v1.services.node_group_controller.transports.NodeGroupControllerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = NodeGroupControllerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport, 'grpc', grpc_helpers), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_node_group_controller_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [node_groups.CreateNodeGroupRequest, dict])
def test_create_node_group(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_node_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.CreateNodeGroupRequest()
    assert isinstance(response, future.Future)

def test_create_node_group_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        client.create_node_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.CreateNodeGroupRequest()

@pytest.mark.asyncio
async def test_create_node_group_async(transport: str='grpc_asyncio', request_type=node_groups.CreateNodeGroupRequest):
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_node_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.CreateNodeGroupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_node_group_async_from_dict():
    await test_create_node_group_async(request_type=dict)

def test_create_node_group_field_headers():
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = node_groups.CreateNodeGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_node_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_node_group_field_headers_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = node_groups.CreateNodeGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_node_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_node_group_flattened():
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_node_group(parent='parent_value', node_group=clusters.NodeGroup(name='name_value'), node_group_id='node_group_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].node_group
        mock_val = clusters.NodeGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].node_group_id
        mock_val = 'node_group_id_value'
        assert arg == mock_val

def test_create_node_group_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_node_group(node_groups.CreateNodeGroupRequest(), parent='parent_value', node_group=clusters.NodeGroup(name='name_value'), node_group_id='node_group_id_value')

@pytest.mark.asyncio
async def test_create_node_group_flattened_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_node_group(parent='parent_value', node_group=clusters.NodeGroup(name='name_value'), node_group_id='node_group_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].node_group
        mock_val = clusters.NodeGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].node_group_id
        mock_val = 'node_group_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_node_group_flattened_error_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_node_group(node_groups.CreateNodeGroupRequest(), parent='parent_value', node_group=clusters.NodeGroup(name='name_value'), node_group_id='node_group_id_value')

@pytest.mark.parametrize('request_type', [node_groups.ResizeNodeGroupRequest, dict])
def test_resize_node_group(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.resize_node_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.ResizeNodeGroupRequest()
    assert isinstance(response, future.Future)

def test_resize_node_group_empty_call():
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        client.resize_node_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.ResizeNodeGroupRequest()

@pytest.mark.asyncio
async def test_resize_node_group_async(transport: str='grpc_asyncio', request_type=node_groups.ResizeNodeGroupRequest):
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.resize_node_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.ResizeNodeGroupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_resize_node_group_async_from_dict():
    await test_resize_node_group_async(request_type=dict)

def test_resize_node_group_field_headers():
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = node_groups.ResizeNodeGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.resize_node_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resize_node_group_field_headers_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = node_groups.ResizeNodeGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.resize_node_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_resize_node_group_flattened():
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.resize_node_group(name='name_value', size=443)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].size
        mock_val = 443
        assert arg == mock_val

def test_resize_node_group_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resize_node_group(node_groups.ResizeNodeGroupRequest(), name='name_value', size=443)

@pytest.mark.asyncio
async def test_resize_node_group_flattened_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resize_node_group), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.resize_node_group(name='name_value', size=443)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].size
        mock_val = 443
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resize_node_group_flattened_error_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resize_node_group(node_groups.ResizeNodeGroupRequest(), name='name_value', size=443)

@pytest.mark.parametrize('request_type', [node_groups.GetNodeGroupRequest, dict])
def test_get_node_group(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        call.return_value = clusters.NodeGroup(name='name_value', roles=[clusters.NodeGroup.Role.DRIVER])
        response = client.get_node_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.GetNodeGroupRequest()
    assert isinstance(response, clusters.NodeGroup)
    assert response.name == 'name_value'
    assert response.roles == [clusters.NodeGroup.Role.DRIVER]

def test_get_node_group_empty_call():
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        client.get_node_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.GetNodeGroupRequest()

@pytest.mark.asyncio
async def test_get_node_group_async(transport: str='grpc_asyncio', request_type=node_groups.GetNodeGroupRequest):
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(clusters.NodeGroup(name='name_value', roles=[clusters.NodeGroup.Role.DRIVER]))
        response = await client.get_node_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == node_groups.GetNodeGroupRequest()
    assert isinstance(response, clusters.NodeGroup)
    assert response.name == 'name_value'
    assert response.roles == [clusters.NodeGroup.Role.DRIVER]

@pytest.mark.asyncio
async def test_get_node_group_async_from_dict():
    await test_get_node_group_async(request_type=dict)

def test_get_node_group_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = node_groups.GetNodeGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        call.return_value = clusters.NodeGroup()
        client.get_node_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_node_group_field_headers_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = node_groups.GetNodeGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(clusters.NodeGroup())
        await client.get_node_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_node_group_flattened():
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        call.return_value = clusters.NodeGroup()
        client.get_node_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_node_group_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_node_group(node_groups.GetNodeGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_node_group_flattened_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_node_group), '__call__') as call:
        call.return_value = clusters.NodeGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(clusters.NodeGroup())
        response = await client.get_node_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_node_group_flattened_error_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_node_group(node_groups.GetNodeGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [node_groups.CreateNodeGroupRequest, dict])
def test_create_node_group_rest(request_type):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/regions/sample2/clusters/sample3'}
    request_init['node_group'] = {'name': 'name_value', 'roles': [1], 'node_group_config': {'num_instances': 1399, 'instance_names': ['instance_names_value1', 'instance_names_value2'], 'instance_references': [{'instance_name': 'instance_name_value', 'instance_id': 'instance_id_value', 'public_key': 'public_key_value', 'public_ecies_key': 'public_ecies_key_value'}], 'image_uri': 'image_uri_value', 'machine_type_uri': 'machine_type_uri_value', 'disk_config': {'boot_disk_type': 'boot_disk_type_value', 'boot_disk_size_gb': 1792, 'num_local_ssds': 1494, 'local_ssd_interface': 'local_ssd_interface_value'}, 'is_preemptible': True, 'preemptibility': 1, 'managed_group_config': {'instance_template_name': 'instance_template_name_value', 'instance_group_manager_name': 'instance_group_manager_name_value', 'instance_group_manager_uri': 'instance_group_manager_uri_value'}, 'accelerators': [{'accelerator_type_uri': 'accelerator_type_uri_value', 'accelerator_count': 1805}], 'min_cpu_platform': 'min_cpu_platform_value', 'min_num_instances': 1818, 'instance_flexibility_policy': {'instance_selection_list': [{'machine_types': ['machine_types_value1', 'machine_types_value2'], 'rank': 428}], 'instance_selection_results': [{'machine_type': 'machine_type_value', 'vm_count': 875}]}, 'startup_config': {'required_registration_fraction': 0.3216}}, 'labels': {}}
    test_field = node_groups.CreateNodeGroupRequest.meta.fields['node_group']

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
    for (field, value) in request_init['node_group'].items():
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
                for i in range(0, len(request_init['node_group'][field])):
                    del request_init['node_group'][field][i][subfield]
            else:
                del request_init['node_group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_node_group(request)
    assert response.operation.name == 'operations/spam'

def test_create_node_group_rest_required_fields(request_type=node_groups.CreateNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupControllerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_node_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_node_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('node_group_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_node_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_node_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_node_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('nodeGroupId', 'requestId')) & set(('parent', 'nodeGroup'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_node_group_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupControllerRestInterceptor())
    client = NodeGroupControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NodeGroupControllerRestInterceptor, 'post_create_node_group') as post, mock.patch.object(transports.NodeGroupControllerRestInterceptor, 'pre_create_node_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = node_groups.CreateNodeGroupRequest.pb(node_groups.CreateNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = node_groups.CreateNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_node_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_node_group_rest_bad_request(transport: str='rest', request_type=node_groups.CreateNodeGroupRequest):
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/regions/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_node_group(request)

def test_create_node_group_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/regions/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value', node_group=clusters.NodeGroup(name='name_value'), node_group_id='node_group_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_node_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/regions/*/clusters/*}/nodeGroups' % client.transport._host, args[1])

def test_create_node_group_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_node_group(node_groups.CreateNodeGroupRequest(), parent='parent_value', node_group=clusters.NodeGroup(name='name_value'), node_group_id='node_group_id_value')

def test_create_node_group_rest_error():
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [node_groups.ResizeNodeGroupRequest, dict])
def test_resize_node_group_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/regions/sample2/clusters/sample3/nodeGroups/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resize_node_group(request)
    assert response.operation.name == 'operations/spam'

def test_resize_node_group_rest_required_fields(request_type=node_groups.ResizeNodeGroupRequest):
    if False:
        return 10
    transport_class = transports.NodeGroupControllerRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['size'] = 0
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize_node_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['size'] = 443
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize_node_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'size' in jsonified_request
    assert jsonified_request['size'] == 443
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resize_node_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resize_node_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resize_node_group._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'size'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resize_node_group_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupControllerRestInterceptor())
    client = NodeGroupControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NodeGroupControllerRestInterceptor, 'post_resize_node_group') as post, mock.patch.object(transports.NodeGroupControllerRestInterceptor, 'pre_resize_node_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = node_groups.ResizeNodeGroupRequest.pb(node_groups.ResizeNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = node_groups.ResizeNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.resize_node_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resize_node_group_rest_bad_request(transport: str='rest', request_type=node_groups.ResizeNodeGroupRequest):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/regions/sample2/clusters/sample3/nodeGroups/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resize_node_group(request)

def test_resize_node_group_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/regions/sample2/clusters/sample3/nodeGroups/sample4'}
        mock_args = dict(name='name_value', size=443)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resize_node_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/regions/*/clusters/*/nodeGroups/*}:resize' % client.transport._host, args[1])

def test_resize_node_group_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resize_node_group(node_groups.ResizeNodeGroupRequest(), name='name_value', size=443)

def test_resize_node_group_rest_error():
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [node_groups.GetNodeGroupRequest, dict])
def test_get_node_group_rest(request_type):
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/regions/sample2/clusters/sample3/nodeGroups/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = clusters.NodeGroup(name='name_value', roles=[clusters.NodeGroup.Role.DRIVER])
        response_value = Response()
        response_value.status_code = 200
        return_value = clusters.NodeGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_node_group(request)
    assert isinstance(response, clusters.NodeGroup)
    assert response.name == 'name_value'
    assert response.roles == [clusters.NodeGroup.Role.DRIVER]

def test_get_node_group_rest_required_fields(request_type=node_groups.GetNodeGroupRequest):
    if False:
        return 10
    transport_class = transports.NodeGroupControllerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_node_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_node_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = clusters.NodeGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = clusters.NodeGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_node_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_node_group_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_node_group._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_node_group_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NodeGroupControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupControllerRestInterceptor())
    client = NodeGroupControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupControllerRestInterceptor, 'post_get_node_group') as post, mock.patch.object(transports.NodeGroupControllerRestInterceptor, 'pre_get_node_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = node_groups.GetNodeGroupRequest.pb(node_groups.GetNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = clusters.NodeGroup.to_json(clusters.NodeGroup())
        request = node_groups.GetNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = clusters.NodeGroup()
        client.get_node_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_node_group_rest_bad_request(transport: str='rest', request_type=node_groups.GetNodeGroupRequest):
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/regions/sample2/clusters/sample3/nodeGroups/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_node_group(request)

def test_get_node_group_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = clusters.NodeGroup()
        sample_request = {'name': 'projects/sample1/regions/sample2/clusters/sample3/nodeGroups/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = clusters.NodeGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_node_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/regions/*/clusters/*/nodeGroups/*}' % client.transport._host, args[1])

def test_get_node_group_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_node_group(node_groups.GetNodeGroupRequest(), name='name_value')

def test_get_node_group_rest_error():
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.NodeGroupControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NodeGroupControllerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.NodeGroupControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NodeGroupControllerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NodeGroupControllerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.NodeGroupControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NodeGroupControllerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.NodeGroupControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = NodeGroupControllerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.NodeGroupControllerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerGrpcAsyncIOTransport, transports.NodeGroupControllerRestTransport])
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
        while True:
            i = 10
    transport = NodeGroupControllerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.NodeGroupControllerGrpcTransport)

def test_node_group_controller_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.NodeGroupControllerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_node_group_controller_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dataproc_v1.services.node_group_controller.transports.NodeGroupControllerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.NodeGroupControllerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_node_group', 'resize_node_group', 'get_node_group', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_node_group_controller_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataproc_v1.services.node_group_controller.transports.NodeGroupControllerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NodeGroupControllerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_node_group_controller_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataproc_v1.services.node_group_controller.transports.NodeGroupControllerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NodeGroupControllerTransport()
        adc.assert_called_once()

def test_node_group_controller_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        NodeGroupControllerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerGrpcAsyncIOTransport])
def test_node_group_controller_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerGrpcAsyncIOTransport, transports.NodeGroupControllerRestTransport])
def test_node_group_controller_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.NodeGroupControllerGrpcTransport, grpc_helpers), (transports.NodeGroupControllerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_node_group_controller_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerGrpcAsyncIOTransport])
def test_node_group_controller_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_node_group_controller_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.NodeGroupControllerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_node_group_controller_rest_lro_client():
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_node_group_controller_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_node_group_controller_host_with_port(transport_name):
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_node_group_controller_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = NodeGroupControllerClient(credentials=creds1, transport=transport_name)
    client2 = NodeGroupControllerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_node_group._session
    session2 = client2.transport.create_node_group._session
    assert session1 != session2
    session1 = client1.transport.resize_node_group._session
    session2 = client2.transport.resize_node_group._session
    assert session1 != session2
    session1 = client1.transport.get_node_group._session
    session2 = client2.transport.get_node_group._session
    assert session1 != session2

def test_node_group_controller_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NodeGroupControllerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_node_group_controller_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NodeGroupControllerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerGrpcAsyncIOTransport])
def test_node_group_controller_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.NodeGroupControllerGrpcTransport, transports.NodeGroupControllerGrpcAsyncIOTransport])
def test_node_group_controller_transport_channel_mtls_with_adc(transport_class):
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

def test_node_group_controller_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_node_group_controller_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_node_group_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    region = 'clam'
    cluster = 'whelk'
    node_group = 'octopus'
    expected = 'projects/{project}/regions/{region}/clusters/{cluster}/nodeGroups/{node_group}'.format(project=project, region=region, cluster=cluster, node_group=node_group)
    actual = NodeGroupControllerClient.node_group_path(project, region, cluster, node_group)
    assert expected == actual

def test_parse_node_group_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'region': 'nudibranch', 'cluster': 'cuttlefish', 'node_group': 'mussel'}
    path = NodeGroupControllerClient.node_group_path(**expected)
    actual = NodeGroupControllerClient.parse_node_group_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = NodeGroupControllerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nautilus'}
    path = NodeGroupControllerClient.common_billing_account_path(**expected)
    actual = NodeGroupControllerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = NodeGroupControllerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'abalone'}
    path = NodeGroupControllerClient.common_folder_path(**expected)
    actual = NodeGroupControllerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = NodeGroupControllerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'clam'}
    path = NodeGroupControllerClient.common_organization_path(**expected)
    actual = NodeGroupControllerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = NodeGroupControllerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus'}
    path = NodeGroupControllerClient.common_project_path(**expected)
    actual = NodeGroupControllerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = NodeGroupControllerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = NodeGroupControllerClient.common_location_path(**expected)
    actual = NodeGroupControllerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.NodeGroupControllerTransport, '_prep_wrapped_messages') as prep:
        client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.NodeGroupControllerTransport, '_prep_wrapped_messages') as prep:
        transport_class = NodeGroupControllerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = NodeGroupControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = NodeGroupControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(NodeGroupControllerClient, transports.NodeGroupControllerGrpcTransport), (NodeGroupControllerAsyncClient, transports.NodeGroupControllerGrpcAsyncIOTransport)])
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