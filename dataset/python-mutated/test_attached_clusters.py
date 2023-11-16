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
from google.cloud.gke_multicloud_v1.services.attached_clusters import AttachedClustersAsyncClient, AttachedClustersClient, pagers, transports
from google.cloud.gke_multicloud_v1.types import attached_resources, attached_service, common_resources

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert AttachedClustersClient._get_default_mtls_endpoint(None) is None
    assert AttachedClustersClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AttachedClustersClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AttachedClustersClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AttachedClustersClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AttachedClustersClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AttachedClustersClient, 'grpc'), (AttachedClustersAsyncClient, 'grpc_asyncio'), (AttachedClustersClient, 'rest')])
def test_attached_clusters_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('gkemulticloud.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AttachedClustersGrpcTransport, 'grpc'), (transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AttachedClustersRestTransport, 'rest')])
def test_attached_clusters_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AttachedClustersClient, 'grpc'), (AttachedClustersAsyncClient, 'grpc_asyncio'), (AttachedClustersClient, 'rest')])
def test_attached_clusters_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('gkemulticloud.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com')

def test_attached_clusters_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = AttachedClustersClient.get_transport_class()
    available_transports = [transports.AttachedClustersGrpcTransport, transports.AttachedClustersRestTransport]
    assert transport in available_transports
    transport = AttachedClustersClient.get_transport_class('grpc')
    assert transport == transports.AttachedClustersGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AttachedClustersClient, transports.AttachedClustersGrpcTransport, 'grpc'), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio'), (AttachedClustersClient, transports.AttachedClustersRestTransport, 'rest')])
@mock.patch.object(AttachedClustersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AttachedClustersClient))
@mock.patch.object(AttachedClustersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AttachedClustersAsyncClient))
def test_attached_clusters_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(AttachedClustersClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AttachedClustersClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AttachedClustersClient, transports.AttachedClustersGrpcTransport, 'grpc', 'true'), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AttachedClustersClient, transports.AttachedClustersGrpcTransport, 'grpc', 'false'), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AttachedClustersClient, transports.AttachedClustersRestTransport, 'rest', 'true'), (AttachedClustersClient, transports.AttachedClustersRestTransport, 'rest', 'false')])
@mock.patch.object(AttachedClustersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AttachedClustersClient))
@mock.patch.object(AttachedClustersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AttachedClustersAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_attached_clusters_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AttachedClustersClient, AttachedClustersAsyncClient])
@mock.patch.object(AttachedClustersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AttachedClustersClient))
@mock.patch.object(AttachedClustersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AttachedClustersAsyncClient))
def test_attached_clusters_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AttachedClustersClient, transports.AttachedClustersGrpcTransport, 'grpc'), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio'), (AttachedClustersClient, transports.AttachedClustersRestTransport, 'rest')])
def test_attached_clusters_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AttachedClustersClient, transports.AttachedClustersGrpcTransport, 'grpc', grpc_helpers), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AttachedClustersClient, transports.AttachedClustersRestTransport, 'rest', None)])
def test_attached_clusters_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_attached_clusters_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.gke_multicloud_v1.services.attached_clusters.transports.AttachedClustersGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AttachedClustersClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AttachedClustersClient, transports.AttachedClustersGrpcTransport, 'grpc', grpc_helpers), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_attached_clusters_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('gkemulticloud.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='gkemulticloud.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [attached_service.CreateAttachedClusterRequest, dict])
def test_create_attached_cluster(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.CreateAttachedClusterRequest()
    assert isinstance(response, future.Future)

def test_create_attached_cluster_empty_call():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        client.create_attached_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.CreateAttachedClusterRequest()

@pytest.mark.asyncio
async def test_create_attached_cluster_async(transport: str='grpc_asyncio', request_type=attached_service.CreateAttachedClusterRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.CreateAttachedClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_attached_cluster_async_from_dict():
    await test_create_attached_cluster_async(request_type=dict)

def test_create_attached_cluster_field_headers():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.CreateAttachedClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_attached_cluster_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.CreateAttachedClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_attached_cluster_flattened():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_attached_cluster(parent='parent_value', attached_cluster=attached_resources.AttachedCluster(name='name_value'), attached_cluster_id='attached_cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].attached_cluster
        mock_val = attached_resources.AttachedCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].attached_cluster_id
        mock_val = 'attached_cluster_id_value'
        assert arg == mock_val

def test_create_attached_cluster_flattened_error():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_attached_cluster(attached_service.CreateAttachedClusterRequest(), parent='parent_value', attached_cluster=attached_resources.AttachedCluster(name='name_value'), attached_cluster_id='attached_cluster_id_value')

@pytest.mark.asyncio
async def test_create_attached_cluster_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_attached_cluster(parent='parent_value', attached_cluster=attached_resources.AttachedCluster(name='name_value'), attached_cluster_id='attached_cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].attached_cluster
        mock_val = attached_resources.AttachedCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].attached_cluster_id
        mock_val = 'attached_cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_attached_cluster_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_attached_cluster(attached_service.CreateAttachedClusterRequest(), parent='parent_value', attached_cluster=attached_resources.AttachedCluster(name='name_value'), attached_cluster_id='attached_cluster_id_value')

@pytest.mark.parametrize('request_type', [attached_service.UpdateAttachedClusterRequest, dict])
def test_update_attached_cluster(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.UpdateAttachedClusterRequest()
    assert isinstance(response, future.Future)

def test_update_attached_cluster_empty_call():
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        client.update_attached_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.UpdateAttachedClusterRequest()

@pytest.mark.asyncio
async def test_update_attached_cluster_async(transport: str='grpc_asyncio', request_type=attached_service.UpdateAttachedClusterRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.UpdateAttachedClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_attached_cluster_async_from_dict():
    await test_update_attached_cluster_async(request_type=dict)

def test_update_attached_cluster_field_headers():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.UpdateAttachedClusterRequest()
    request.attached_cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attached_cluster.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_attached_cluster_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.UpdateAttachedClusterRequest()
    request.attached_cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'attached_cluster.name=name_value') in kw['metadata']

def test_update_attached_cluster_flattened():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_attached_cluster(attached_cluster=attached_resources.AttachedCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].attached_cluster
        mock_val = attached_resources.AttachedCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_attached_cluster_flattened_error():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_attached_cluster(attached_service.UpdateAttachedClusterRequest(), attached_cluster=attached_resources.AttachedCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_attached_cluster_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_attached_cluster(attached_cluster=attached_resources.AttachedCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].attached_cluster
        mock_val = attached_resources.AttachedCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_attached_cluster_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_attached_cluster(attached_service.UpdateAttachedClusterRequest(), attached_cluster=attached_resources.AttachedCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [attached_service.ImportAttachedClusterRequest, dict])
def test_import_attached_cluster(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.import_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.ImportAttachedClusterRequest()
    assert isinstance(response, future.Future)

def test_import_attached_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        client.import_attached_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.ImportAttachedClusterRequest()

@pytest.mark.asyncio
async def test_import_attached_cluster_async(transport: str='grpc_asyncio', request_type=attached_service.ImportAttachedClusterRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.ImportAttachedClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_import_attached_cluster_async_from_dict():
    await test_import_attached_cluster_async(request_type=dict)

def test_import_attached_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.ImportAttachedClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_attached_cluster_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.ImportAttachedClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.import_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_import_attached_cluster_flattened():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_attached_cluster(parent='parent_value', fleet_membership='fleet_membership_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].fleet_membership
        mock_val = 'fleet_membership_value'
        assert arg == mock_val

def test_import_attached_cluster_flattened_error():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.import_attached_cluster(attached_service.ImportAttachedClusterRequest(), parent='parent_value', fleet_membership='fleet_membership_value')

@pytest.mark.asyncio
async def test_import_attached_cluster_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_attached_cluster(parent='parent_value', fleet_membership='fleet_membership_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].fleet_membership
        mock_val = 'fleet_membership_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_import_attached_cluster_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.import_attached_cluster(attached_service.ImportAttachedClusterRequest(), parent='parent_value', fleet_membership='fleet_membership_value')

@pytest.mark.parametrize('request_type', [attached_service.GetAttachedClusterRequest, dict])
def test_get_attached_cluster(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        call.return_value = attached_resources.AttachedCluster(name='name_value', description='description_value', platform_version='platform_version_value', distribution='distribution_value', cluster_region='cluster_region_value', state=attached_resources.AttachedCluster.State.PROVISIONING, uid='uid_value', reconciling=True, etag='etag_value', kubernetes_version='kubernetes_version_value')
        response = client.get_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GetAttachedClusterRequest()
    assert isinstance(response, attached_resources.AttachedCluster)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.platform_version == 'platform_version_value'
    assert response.distribution == 'distribution_value'
    assert response.cluster_region == 'cluster_region_value'
    assert response.state == attached_resources.AttachedCluster.State.PROVISIONING
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.kubernetes_version == 'kubernetes_version_value'

def test_get_attached_cluster_empty_call():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        client.get_attached_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GetAttachedClusterRequest()

@pytest.mark.asyncio
async def test_get_attached_cluster_async(transport: str='grpc_asyncio', request_type=attached_service.GetAttachedClusterRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_resources.AttachedCluster(name='name_value', description='description_value', platform_version='platform_version_value', distribution='distribution_value', cluster_region='cluster_region_value', state=attached_resources.AttachedCluster.State.PROVISIONING, uid='uid_value', reconciling=True, etag='etag_value', kubernetes_version='kubernetes_version_value'))
        response = await client.get_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GetAttachedClusterRequest()
    assert isinstance(response, attached_resources.AttachedCluster)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.platform_version == 'platform_version_value'
    assert response.distribution == 'distribution_value'
    assert response.cluster_region == 'cluster_region_value'
    assert response.state == attached_resources.AttachedCluster.State.PROVISIONING
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.kubernetes_version == 'kubernetes_version_value'

@pytest.mark.asyncio
async def test_get_attached_cluster_async_from_dict():
    await test_get_attached_cluster_async(request_type=dict)

def test_get_attached_cluster_field_headers():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.GetAttachedClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        call.return_value = attached_resources.AttachedCluster()
        client.get_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_attached_cluster_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.GetAttachedClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_resources.AttachedCluster())
        await client.get_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_attached_cluster_flattened():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        call.return_value = attached_resources.AttachedCluster()
        client.get_attached_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_attached_cluster_flattened_error():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_attached_cluster(attached_service.GetAttachedClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_attached_cluster_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_attached_cluster), '__call__') as call:
        call.return_value = attached_resources.AttachedCluster()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_resources.AttachedCluster())
        response = await client.get_attached_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_attached_cluster_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_attached_cluster(attached_service.GetAttachedClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [attached_service.ListAttachedClustersRequest, dict])
def test_list_attached_clusters(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.return_value = attached_service.ListAttachedClustersResponse(next_page_token='next_page_token_value')
        response = client.list_attached_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.ListAttachedClustersRequest()
    assert isinstance(response, pagers.ListAttachedClustersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_attached_clusters_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        client.list_attached_clusters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.ListAttachedClustersRequest()

@pytest.mark.asyncio
async def test_list_attached_clusters_async(transport: str='grpc_asyncio', request_type=attached_service.ListAttachedClustersRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_service.ListAttachedClustersResponse(next_page_token='next_page_token_value'))
        response = await client.list_attached_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.ListAttachedClustersRequest()
    assert isinstance(response, pagers.ListAttachedClustersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_attached_clusters_async_from_dict():
    await test_list_attached_clusters_async(request_type=dict)

def test_list_attached_clusters_field_headers():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.ListAttachedClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.return_value = attached_service.ListAttachedClustersResponse()
        client.list_attached_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_attached_clusters_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.ListAttachedClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_service.ListAttachedClustersResponse())
        await client.list_attached_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_attached_clusters_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.return_value = attached_service.ListAttachedClustersResponse()
        client.list_attached_clusters(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_attached_clusters_flattened_error():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_attached_clusters(attached_service.ListAttachedClustersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_attached_clusters_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.return_value = attached_service.ListAttachedClustersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_service.ListAttachedClustersResponse())
        response = await client.list_attached_clusters(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_attached_clusters_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_attached_clusters(attached_service.ListAttachedClustersRequest(), parent='parent_value')

def test_list_attached_clusters_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.side_effect = (attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster(), attached_resources.AttachedCluster()], next_page_token='abc'), attached_service.ListAttachedClustersResponse(attached_clusters=[], next_page_token='def'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster()], next_page_token='ghi'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_attached_clusters(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, attached_resources.AttachedCluster) for i in results))

def test_list_attached_clusters_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__') as call:
        call.side_effect = (attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster(), attached_resources.AttachedCluster()], next_page_token='abc'), attached_service.ListAttachedClustersResponse(attached_clusters=[], next_page_token='def'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster()], next_page_token='ghi'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster()]), RuntimeError)
        pages = list(client.list_attached_clusters(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_attached_clusters_async_pager():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster(), attached_resources.AttachedCluster()], next_page_token='abc'), attached_service.ListAttachedClustersResponse(attached_clusters=[], next_page_token='def'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster()], next_page_token='ghi'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster()]), RuntimeError)
        async_pager = await client.list_attached_clusters(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, attached_resources.AttachedCluster) for i in responses))

@pytest.mark.asyncio
async def test_list_attached_clusters_async_pages():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_attached_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster(), attached_resources.AttachedCluster()], next_page_token='abc'), attached_service.ListAttachedClustersResponse(attached_clusters=[], next_page_token='def'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster()], next_page_token='ghi'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_attached_clusters(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [attached_service.DeleteAttachedClusterRequest, dict])
def test_delete_attached_cluster(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.DeleteAttachedClusterRequest()
    assert isinstance(response, future.Future)

def test_delete_attached_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        client.delete_attached_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.DeleteAttachedClusterRequest()

@pytest.mark.asyncio
async def test_delete_attached_cluster_async(transport: str='grpc_asyncio', request_type=attached_service.DeleteAttachedClusterRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.DeleteAttachedClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_attached_cluster_async_from_dict():
    await test_delete_attached_cluster_async(request_type=dict)

def test_delete_attached_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.DeleteAttachedClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_attached_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_attached_cluster_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.DeleteAttachedClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_attached_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_attached_cluster_flattened():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_attached_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_attached_cluster_flattened_error():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_attached_cluster(attached_service.DeleteAttachedClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_attached_cluster_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_attached_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_attached_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_attached_cluster_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_attached_cluster(attached_service.DeleteAttachedClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [attached_service.GetAttachedServerConfigRequest, dict])
def test_get_attached_server_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        call.return_value = attached_resources.AttachedServerConfig(name='name_value')
        response = client.get_attached_server_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GetAttachedServerConfigRequest()
    assert isinstance(response, attached_resources.AttachedServerConfig)
    assert response.name == 'name_value'

def test_get_attached_server_config_empty_call():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        client.get_attached_server_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GetAttachedServerConfigRequest()

@pytest.mark.asyncio
async def test_get_attached_server_config_async(transport: str='grpc_asyncio', request_type=attached_service.GetAttachedServerConfigRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_resources.AttachedServerConfig(name='name_value'))
        response = await client.get_attached_server_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GetAttachedServerConfigRequest()
    assert isinstance(response, attached_resources.AttachedServerConfig)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_attached_server_config_async_from_dict():
    await test_get_attached_server_config_async(request_type=dict)

def test_get_attached_server_config_field_headers():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.GetAttachedServerConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        call.return_value = attached_resources.AttachedServerConfig()
        client.get_attached_server_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_attached_server_config_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.GetAttachedServerConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_resources.AttachedServerConfig())
        await client.get_attached_server_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_attached_server_config_flattened():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        call.return_value = attached_resources.AttachedServerConfig()
        client.get_attached_server_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_attached_server_config_flattened_error():
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_attached_server_config(attached_service.GetAttachedServerConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_attached_server_config_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_attached_server_config), '__call__') as call:
        call.return_value = attached_resources.AttachedServerConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_resources.AttachedServerConfig())
        response = await client.get_attached_server_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_attached_server_config_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_attached_server_config(attached_service.GetAttachedServerConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [attached_service.GenerateAttachedClusterInstallManifestRequest, dict])
def test_generate_attached_cluster_install_manifest(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        call.return_value = attached_service.GenerateAttachedClusterInstallManifestResponse(manifest='manifest_value')
        response = client.generate_attached_cluster_install_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GenerateAttachedClusterInstallManifestRequest()
    assert isinstance(response, attached_service.GenerateAttachedClusterInstallManifestResponse)
    assert response.manifest == 'manifest_value'

def test_generate_attached_cluster_install_manifest_empty_call():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        client.generate_attached_cluster_install_manifest()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GenerateAttachedClusterInstallManifestRequest()

@pytest.mark.asyncio
async def test_generate_attached_cluster_install_manifest_async(transport: str='grpc_asyncio', request_type=attached_service.GenerateAttachedClusterInstallManifestRequest):
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_service.GenerateAttachedClusterInstallManifestResponse(manifest='manifest_value'))
        response = await client.generate_attached_cluster_install_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == attached_service.GenerateAttachedClusterInstallManifestRequest()
    assert isinstance(response, attached_service.GenerateAttachedClusterInstallManifestResponse)
    assert response.manifest == 'manifest_value'

@pytest.mark.asyncio
async def test_generate_attached_cluster_install_manifest_async_from_dict():
    await test_generate_attached_cluster_install_manifest_async(request_type=dict)

def test_generate_attached_cluster_install_manifest_field_headers():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.GenerateAttachedClusterInstallManifestRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        call.return_value = attached_service.GenerateAttachedClusterInstallManifestResponse()
        client.generate_attached_cluster_install_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_attached_cluster_install_manifest_field_headers_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = attached_service.GenerateAttachedClusterInstallManifestRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_service.GenerateAttachedClusterInstallManifestResponse())
        await client.generate_attached_cluster_install_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_generate_attached_cluster_install_manifest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        call.return_value = attached_service.GenerateAttachedClusterInstallManifestResponse()
        client.generate_attached_cluster_install_manifest(parent='parent_value', attached_cluster_id='attached_cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].attached_cluster_id
        mock_val = 'attached_cluster_id_value'
        assert arg == mock_val

def test_generate_attached_cluster_install_manifest_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.generate_attached_cluster_install_manifest(attached_service.GenerateAttachedClusterInstallManifestRequest(), parent='parent_value', attached_cluster_id='attached_cluster_id_value')

@pytest.mark.asyncio
async def test_generate_attached_cluster_install_manifest_flattened_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_attached_cluster_install_manifest), '__call__') as call:
        call.return_value = attached_service.GenerateAttachedClusterInstallManifestResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(attached_service.GenerateAttachedClusterInstallManifestResponse())
        response = await client.generate_attached_cluster_install_manifest(parent='parent_value', attached_cluster_id='attached_cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].attached_cluster_id
        mock_val = 'attached_cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_generate_attached_cluster_install_manifest_flattened_error_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.generate_attached_cluster_install_manifest(attached_service.GenerateAttachedClusterInstallManifestRequest(), parent='parent_value', attached_cluster_id='attached_cluster_id_value')

@pytest.mark.parametrize('request_type', [attached_service.CreateAttachedClusterRequest, dict])
def test_create_attached_cluster_rest(request_type):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['attached_cluster'] = {'name': 'name_value', 'description': 'description_value', 'oidc_config': {'issuer_url': 'issuer_url_value', 'jwks': b'jwks_blob'}, 'platform_version': 'platform_version_value', 'distribution': 'distribution_value', 'cluster_region': 'cluster_region_value', 'fleet': {'project': 'project_value', 'membership': 'membership_value'}, 'state': 1, 'uid': 'uid_value', 'reconciling': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'kubernetes_version': 'kubernetes_version_value', 'annotations': {}, 'workload_identity_config': {'issuer_uri': 'issuer_uri_value', 'workload_pool': 'workload_pool_value', 'identity_provider': 'identity_provider_value'}, 'logging_config': {'component_config': {'enable_components': [1]}}, 'errors': [{'message': 'message_value'}], 'authorization': {'admin_users': [{'username': 'username_value'}]}, 'monitoring_config': {'managed_prometheus_config': {'enabled': True}}}
    test_field = attached_service.CreateAttachedClusterRequest.meta.fields['attached_cluster']

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
    for (field, value) in request_init['attached_cluster'].items():
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
                for i in range(0, len(request_init['attached_cluster'][field])):
                    del request_init['attached_cluster'][field][i][subfield]
            else:
                del request_init['attached_cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_attached_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_create_attached_cluster_rest_required_fields(request_type=attached_service.CreateAttachedClusterRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['attached_cluster_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'attachedClusterId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'attachedClusterId' in jsonified_request
    assert jsonified_request['attachedClusterId'] == request_init['attached_cluster_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['attachedClusterId'] = 'attached_cluster_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_attached_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('attached_cluster_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'attachedClusterId' in jsonified_request
    assert jsonified_request['attachedClusterId'] == 'attached_cluster_id_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_attached_cluster(request)
            expected_params = [('attachedClusterId', '')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_attached_cluster_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_attached_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('attachedClusterId', 'validateOnly')) & set(('parent', 'attachedCluster', 'attachedClusterId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_attached_cluster_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_create_attached_cluster') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_create_attached_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.CreateAttachedClusterRequest.pb(attached_service.CreateAttachedClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = attached_service.CreateAttachedClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_attached_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_attached_cluster_rest_bad_request(transport: str='rest', request_type=attached_service.CreateAttachedClusterRequest):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_attached_cluster(request)

def test_create_attached_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', attached_cluster=attached_resources.AttachedCluster(name='name_value'), attached_cluster_id='attached_cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_attached_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/attachedClusters' % client.transport._host, args[1])

def test_create_attached_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_attached_cluster(attached_service.CreateAttachedClusterRequest(), parent='parent_value', attached_cluster=attached_resources.AttachedCluster(name='name_value'), attached_cluster_id='attached_cluster_id_value')

def test_create_attached_cluster_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [attached_service.UpdateAttachedClusterRequest, dict])
def test_update_attached_cluster_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'attached_cluster': {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}}
    request_init['attached_cluster'] = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3', 'description': 'description_value', 'oidc_config': {'issuer_url': 'issuer_url_value', 'jwks': b'jwks_blob'}, 'platform_version': 'platform_version_value', 'distribution': 'distribution_value', 'cluster_region': 'cluster_region_value', 'fleet': {'project': 'project_value', 'membership': 'membership_value'}, 'state': 1, 'uid': 'uid_value', 'reconciling': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'kubernetes_version': 'kubernetes_version_value', 'annotations': {}, 'workload_identity_config': {'issuer_uri': 'issuer_uri_value', 'workload_pool': 'workload_pool_value', 'identity_provider': 'identity_provider_value'}, 'logging_config': {'component_config': {'enable_components': [1]}}, 'errors': [{'message': 'message_value'}], 'authorization': {'admin_users': [{'username': 'username_value'}]}, 'monitoring_config': {'managed_prometheus_config': {'enabled': True}}}
    test_field = attached_service.UpdateAttachedClusterRequest.meta.fields['attached_cluster']

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
    for (field, value) in request_init['attached_cluster'].items():
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
                for i in range(0, len(request_init['attached_cluster'][field])):
                    del request_init['attached_cluster'][field][i][subfield]
            else:
                del request_init['attached_cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_attached_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_update_attached_cluster_rest_required_fields(request_type=attached_service.UpdateAttachedClusterRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_attached_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_attached_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_attached_cluster_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_attached_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask', 'validateOnly')) & set(('attachedCluster', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_attached_cluster_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_update_attached_cluster') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_update_attached_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.UpdateAttachedClusterRequest.pb(attached_service.UpdateAttachedClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = attached_service.UpdateAttachedClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_attached_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_attached_cluster_rest_bad_request(transport: str='rest', request_type=attached_service.UpdateAttachedClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'attached_cluster': {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_attached_cluster(request)

def test_update_attached_cluster_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'attached_cluster': {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}}
        mock_args = dict(attached_cluster=attached_resources.AttachedCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_attached_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{attached_cluster.name=projects/*/locations/*/attachedClusters/*}' % client.transport._host, args[1])

def test_update_attached_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_attached_cluster(attached_service.UpdateAttachedClusterRequest(), attached_cluster=attached_resources.AttachedCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_attached_cluster_rest_error():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [attached_service.ImportAttachedClusterRequest, dict])
def test_import_attached_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_attached_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_import_attached_cluster_rest_required_fields(request_type=attached_service.ImportAttachedClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['fleet_membership'] = ''
    request_init['platform_version'] = ''
    request_init['distribution'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['fleetMembership'] = 'fleet_membership_value'
    jsonified_request['platformVersion'] = 'platform_version_value'
    jsonified_request['distribution'] = 'distribution_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'fleetMembership' in jsonified_request
    assert jsonified_request['fleetMembership'] == 'fleet_membership_value'
    assert 'platformVersion' in jsonified_request
    assert jsonified_request['platformVersion'] == 'platform_version_value'
    assert 'distribution' in jsonified_request
    assert jsonified_request['distribution'] == 'distribution_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.import_attached_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_attached_cluster_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_attached_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'fleetMembership', 'platformVersion', 'distribution'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_attached_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_import_attached_cluster') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_import_attached_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.ImportAttachedClusterRequest.pb(attached_service.ImportAttachedClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = attached_service.ImportAttachedClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.import_attached_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_attached_cluster_rest_bad_request(transport: str='rest', request_type=attached_service.ImportAttachedClusterRequest):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_attached_cluster(request)

def test_import_attached_cluster_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', fleet_membership='fleet_membership_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.import_attached_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/attachedClusters:import' % client.transport._host, args[1])

def test_import_attached_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.import_attached_cluster(attached_service.ImportAttachedClusterRequest(), parent='parent_value', fleet_membership='fleet_membership_value')

def test_import_attached_cluster_rest_error():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [attached_service.GetAttachedClusterRequest, dict])
def test_get_attached_cluster_rest(request_type):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_resources.AttachedCluster(name='name_value', description='description_value', platform_version='platform_version_value', distribution='distribution_value', cluster_region='cluster_region_value', state=attached_resources.AttachedCluster.State.PROVISIONING, uid='uid_value', reconciling=True, etag='etag_value', kubernetes_version='kubernetes_version_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_resources.AttachedCluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_attached_cluster(request)
    assert isinstance(response, attached_resources.AttachedCluster)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.platform_version == 'platform_version_value'
    assert response.distribution == 'distribution_value'
    assert response.cluster_region == 'cluster_region_value'
    assert response.state == attached_resources.AttachedCluster.State.PROVISIONING
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.kubernetes_version == 'kubernetes_version_value'

def test_get_attached_cluster_rest_required_fields(request_type=attached_service.GetAttachedClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = attached_resources.AttachedCluster()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = attached_resources.AttachedCluster.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_attached_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_attached_cluster_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_attached_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_attached_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_get_attached_cluster') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_get_attached_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.GetAttachedClusterRequest.pb(attached_service.GetAttachedClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = attached_resources.AttachedCluster.to_json(attached_resources.AttachedCluster())
        request = attached_service.GetAttachedClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = attached_resources.AttachedCluster()
        client.get_attached_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_attached_cluster_rest_bad_request(transport: str='rest', request_type=attached_service.GetAttachedClusterRequest):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_attached_cluster(request)

def test_get_attached_cluster_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_resources.AttachedCluster()
        sample_request = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_resources.AttachedCluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_attached_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/attachedClusters/*}' % client.transport._host, args[1])

def test_get_attached_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_attached_cluster(attached_service.GetAttachedClusterRequest(), name='name_value')

def test_get_attached_cluster_rest_error():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [attached_service.ListAttachedClustersRequest, dict])
def test_list_attached_clusters_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_service.ListAttachedClustersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_service.ListAttachedClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_attached_clusters(request)
    assert isinstance(response, pagers.ListAttachedClustersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_attached_clusters_rest_required_fields(request_type=attached_service.ListAttachedClustersRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_attached_clusters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_attached_clusters._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = attached_service.ListAttachedClustersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = attached_service.ListAttachedClustersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_attached_clusters(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_attached_clusters_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_attached_clusters._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_attached_clusters_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_list_attached_clusters') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_list_attached_clusters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.ListAttachedClustersRequest.pb(attached_service.ListAttachedClustersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = attached_service.ListAttachedClustersResponse.to_json(attached_service.ListAttachedClustersResponse())
        request = attached_service.ListAttachedClustersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = attached_service.ListAttachedClustersResponse()
        client.list_attached_clusters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_attached_clusters_rest_bad_request(transport: str='rest', request_type=attached_service.ListAttachedClustersRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_attached_clusters(request)

def test_list_attached_clusters_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_service.ListAttachedClustersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_service.ListAttachedClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_attached_clusters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/attachedClusters' % client.transport._host, args[1])

def test_list_attached_clusters_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_attached_clusters(attached_service.ListAttachedClustersRequest(), parent='parent_value')

def test_list_attached_clusters_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster(), attached_resources.AttachedCluster()], next_page_token='abc'), attached_service.ListAttachedClustersResponse(attached_clusters=[], next_page_token='def'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster()], next_page_token='ghi'), attached_service.ListAttachedClustersResponse(attached_clusters=[attached_resources.AttachedCluster(), attached_resources.AttachedCluster()]))
        response = response + response
        response = tuple((attached_service.ListAttachedClustersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_attached_clusters(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, attached_resources.AttachedCluster) for i in results))
        pages = list(client.list_attached_clusters(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [attached_service.DeleteAttachedClusterRequest, dict])
def test_delete_attached_cluster_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_attached_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_delete_attached_cluster_rest_required_fields(request_type=attached_service.DeleteAttachedClusterRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_attached_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_attached_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'ignore_errors', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_attached_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_attached_cluster_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_attached_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'ignoreErrors', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_attached_cluster_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_delete_attached_cluster') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_delete_attached_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.DeleteAttachedClusterRequest.pb(attached_service.DeleteAttachedClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = attached_service.DeleteAttachedClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_attached_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_attached_cluster_rest_bad_request(transport: str='rest', request_type=attached_service.DeleteAttachedClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_attached_cluster(request)

def test_delete_attached_cluster_rest_flattened():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/attachedClusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_attached_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/attachedClusters/*}' % client.transport._host, args[1])

def test_delete_attached_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_attached_cluster(attached_service.DeleteAttachedClusterRequest(), name='name_value')

def test_delete_attached_cluster_rest_error():
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [attached_service.GetAttachedServerConfigRequest, dict])
def test_get_attached_server_config_rest(request_type):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/attachedServerConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_resources.AttachedServerConfig(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_resources.AttachedServerConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_attached_server_config(request)
    assert isinstance(response, attached_resources.AttachedServerConfig)
    assert response.name == 'name_value'

def test_get_attached_server_config_rest_required_fields(request_type=attached_service.GetAttachedServerConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_attached_server_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_attached_server_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = attached_resources.AttachedServerConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = attached_resources.AttachedServerConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_attached_server_config(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_attached_server_config_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_attached_server_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_attached_server_config_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_get_attached_server_config') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_get_attached_server_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.GetAttachedServerConfigRequest.pb(attached_service.GetAttachedServerConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = attached_resources.AttachedServerConfig.to_json(attached_resources.AttachedServerConfig())
        request = attached_service.GetAttachedServerConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = attached_resources.AttachedServerConfig()
        client.get_attached_server_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_attached_server_config_rest_bad_request(transport: str='rest', request_type=attached_service.GetAttachedServerConfigRequest):
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/attachedServerConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_attached_server_config(request)

def test_get_attached_server_config_rest_flattened():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_resources.AttachedServerConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/attachedServerConfig'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_resources.AttachedServerConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_attached_server_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/attachedServerConfig}' % client.transport._host, args[1])

def test_get_attached_server_config_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_attached_server_config(attached_service.GetAttachedServerConfigRequest(), name='name_value')

def test_get_attached_server_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [attached_service.GenerateAttachedClusterInstallManifestRequest, dict])
def test_generate_attached_cluster_install_manifest_rest(request_type):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_service.GenerateAttachedClusterInstallManifestResponse(manifest='manifest_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_service.GenerateAttachedClusterInstallManifestResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_attached_cluster_install_manifest(request)
    assert isinstance(response, attached_service.GenerateAttachedClusterInstallManifestResponse)
    assert response.manifest == 'manifest_value'

def test_generate_attached_cluster_install_manifest_rest_required_fields(request_type=attached_service.GenerateAttachedClusterInstallManifestRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AttachedClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['attached_cluster_id'] = ''
    request_init['platform_version'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'attachedClusterId' not in jsonified_request
    assert 'platformVersion' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_attached_cluster_install_manifest._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'attachedClusterId' in jsonified_request
    assert jsonified_request['attachedClusterId'] == request_init['attached_cluster_id']
    assert 'platformVersion' in jsonified_request
    assert jsonified_request['platformVersion'] == request_init['platform_version']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['attachedClusterId'] = 'attached_cluster_id_value'
    jsonified_request['platformVersion'] = 'platform_version_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_attached_cluster_install_manifest._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('attached_cluster_id', 'platform_version'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'attachedClusterId' in jsonified_request
    assert jsonified_request['attachedClusterId'] == 'attached_cluster_id_value'
    assert 'platformVersion' in jsonified_request
    assert jsonified_request['platformVersion'] == 'platform_version_value'
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = attached_service.GenerateAttachedClusterInstallManifestResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = attached_service.GenerateAttachedClusterInstallManifestResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_attached_cluster_install_manifest(request)
            expected_params = [('attachedClusterId', ''), ('platformVersion', '')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_attached_cluster_install_manifest_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_attached_cluster_install_manifest._get_unset_required_fields({})
    assert set(unset_fields) == set(('attachedClusterId', 'platformVersion')) & set(('parent', 'attachedClusterId', 'platformVersion'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_attached_cluster_install_manifest_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AttachedClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AttachedClustersRestInterceptor())
    client = AttachedClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AttachedClustersRestInterceptor, 'post_generate_attached_cluster_install_manifest') as post, mock.patch.object(transports.AttachedClustersRestInterceptor, 'pre_generate_attached_cluster_install_manifest') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = attached_service.GenerateAttachedClusterInstallManifestRequest.pb(attached_service.GenerateAttachedClusterInstallManifestRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = attached_service.GenerateAttachedClusterInstallManifestResponse.to_json(attached_service.GenerateAttachedClusterInstallManifestResponse())
        request = attached_service.GenerateAttachedClusterInstallManifestRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = attached_service.GenerateAttachedClusterInstallManifestResponse()
        client.generate_attached_cluster_install_manifest(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_attached_cluster_install_manifest_rest_bad_request(transport: str='rest', request_type=attached_service.GenerateAttachedClusterInstallManifestRequest):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_attached_cluster_install_manifest(request)

def test_generate_attached_cluster_install_manifest_rest_flattened():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = attached_service.GenerateAttachedClusterInstallManifestResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', attached_cluster_id='attached_cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = attached_service.GenerateAttachedClusterInstallManifestResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.generate_attached_cluster_install_manifest(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}:generateAttachedClusterInstallManifest' % client.transport._host, args[1])

def test_generate_attached_cluster_install_manifest_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.generate_attached_cluster_install_manifest(attached_service.GenerateAttachedClusterInstallManifestRequest(), parent='parent_value', attached_cluster_id='attached_cluster_id_value')

def test_generate_attached_cluster_install_manifest_rest_error():
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.AttachedClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AttachedClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AttachedClustersClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AttachedClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AttachedClustersClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AttachedClustersClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AttachedClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AttachedClustersClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AttachedClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AttachedClustersClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AttachedClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AttachedClustersGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AttachedClustersGrpcTransport, transports.AttachedClustersGrpcAsyncIOTransport, transports.AttachedClustersRestTransport])
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
        print('Hello World!')
    transport = AttachedClustersClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AttachedClustersGrpcTransport)

def test_attached_clusters_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AttachedClustersTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_attached_clusters_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.gke_multicloud_v1.services.attached_clusters.transports.AttachedClustersTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AttachedClustersTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_attached_cluster', 'update_attached_cluster', 'import_attached_cluster', 'get_attached_cluster', 'list_attached_clusters', 'delete_attached_cluster', 'get_attached_server_config', 'generate_attached_cluster_install_manifest', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_attached_clusters_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.gke_multicloud_v1.services.attached_clusters.transports.AttachedClustersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AttachedClustersTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_attached_clusters_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.gke_multicloud_v1.services.attached_clusters.transports.AttachedClustersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AttachedClustersTransport()
        adc.assert_called_once()

def test_attached_clusters_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AttachedClustersClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AttachedClustersGrpcTransport, transports.AttachedClustersGrpcAsyncIOTransport])
def test_attached_clusters_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AttachedClustersGrpcTransport, transports.AttachedClustersGrpcAsyncIOTransport, transports.AttachedClustersRestTransport])
def test_attached_clusters_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AttachedClustersGrpcTransport, grpc_helpers), (transports.AttachedClustersGrpcAsyncIOTransport, grpc_helpers_async)])
def test_attached_clusters_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('gkemulticloud.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='gkemulticloud.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AttachedClustersGrpcTransport, transports.AttachedClustersGrpcAsyncIOTransport])
def test_attached_clusters_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_attached_clusters_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AttachedClustersRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_attached_clusters_rest_lro_client():
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_attached_clusters_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkemulticloud.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('gkemulticloud.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_attached_clusters_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkemulticloud.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('gkemulticloud.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_attached_clusters_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AttachedClustersClient(credentials=creds1, transport=transport_name)
    client2 = AttachedClustersClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_attached_cluster._session
    session2 = client2.transport.create_attached_cluster._session
    assert session1 != session2
    session1 = client1.transport.update_attached_cluster._session
    session2 = client2.transport.update_attached_cluster._session
    assert session1 != session2
    session1 = client1.transport.import_attached_cluster._session
    session2 = client2.transport.import_attached_cluster._session
    assert session1 != session2
    session1 = client1.transport.get_attached_cluster._session
    session2 = client2.transport.get_attached_cluster._session
    assert session1 != session2
    session1 = client1.transport.list_attached_clusters._session
    session2 = client2.transport.list_attached_clusters._session
    assert session1 != session2
    session1 = client1.transport.delete_attached_cluster._session
    session2 = client2.transport.delete_attached_cluster._session
    assert session1 != session2
    session1 = client1.transport.get_attached_server_config._session
    session2 = client2.transport.get_attached_server_config._session
    assert session1 != session2
    session1 = client1.transport.generate_attached_cluster_install_manifest._session
    session2 = client2.transport.generate_attached_cluster_install_manifest._session
    assert session1 != session2

def test_attached_clusters_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AttachedClustersGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_attached_clusters_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AttachedClustersGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AttachedClustersGrpcTransport, transports.AttachedClustersGrpcAsyncIOTransport])
def test_attached_clusters_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AttachedClustersGrpcTransport, transports.AttachedClustersGrpcAsyncIOTransport])
def test_attached_clusters_transport_channel_mtls_with_adc(transport_class):
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

def test_attached_clusters_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_attached_clusters_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_attached_cluster_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    attached_cluster = 'whelk'
    expected = 'projects/{project}/locations/{location}/attachedClusters/{attached_cluster}'.format(project=project, location=location, attached_cluster=attached_cluster)
    actual = AttachedClustersClient.attached_cluster_path(project, location, attached_cluster)
    assert expected == actual

def test_parse_attached_cluster_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'attached_cluster': 'nudibranch'}
    path = AttachedClustersClient.attached_cluster_path(**expected)
    actual = AttachedClustersClient.parse_attached_cluster_path(path)
    assert expected == actual

def test_attached_server_config_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}/attachedServerConfig'.format(project=project, location=location)
    actual = AttachedClustersClient.attached_server_config_path(project, location)
    assert expected == actual

def test_parse_attached_server_config_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = AttachedClustersClient.attached_server_config_path(**expected)
    actual = AttachedClustersClient.parse_attached_server_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AttachedClustersClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'abalone'}
    path = AttachedClustersClient.common_billing_account_path(**expected)
    actual = AttachedClustersClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AttachedClustersClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'clam'}
    path = AttachedClustersClient.common_folder_path(**expected)
    actual = AttachedClustersClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AttachedClustersClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'octopus'}
    path = AttachedClustersClient.common_organization_path(**expected)
    actual = AttachedClustersClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = AttachedClustersClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch'}
    path = AttachedClustersClient.common_project_path(**expected)
    actual = AttachedClustersClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AttachedClustersClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = AttachedClustersClient.common_location_path(**expected)
    actual = AttachedClustersClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AttachedClustersTransport, '_prep_wrapped_messages') as prep:
        client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AttachedClustersTransport, '_prep_wrapped_messages') as prep:
        transport_class = AttachedClustersClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
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
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AttachedClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = AttachedClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AttachedClustersClient, transports.AttachedClustersGrpcTransport), (AttachedClustersAsyncClient, transports.AttachedClustersGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)