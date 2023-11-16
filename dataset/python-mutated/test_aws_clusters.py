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
from google.cloud.gke_multicloud_v1.services.aws_clusters import AwsClustersAsyncClient, AwsClustersClient, pagers, transports
from google.cloud.gke_multicloud_v1.types import aws_resources, aws_service, common_resources

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert AwsClustersClient._get_default_mtls_endpoint(None) is None
    assert AwsClustersClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AwsClustersClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AwsClustersClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AwsClustersClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AwsClustersClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AwsClustersClient, 'grpc'), (AwsClustersAsyncClient, 'grpc_asyncio'), (AwsClustersClient, 'rest')])
def test_aws_clusters_client_from_service_account_info(client_class, transport_name):
    if False:
        i = 10
        return i + 15
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('gkemulticloud.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AwsClustersGrpcTransport, 'grpc'), (transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AwsClustersRestTransport, 'rest')])
def test_aws_clusters_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(AwsClustersClient, 'grpc'), (AwsClustersAsyncClient, 'grpc_asyncio'), (AwsClustersClient, 'rest')])
def test_aws_clusters_client_from_service_account_file(client_class, transport_name):
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

def test_aws_clusters_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = AwsClustersClient.get_transport_class()
    available_transports = [transports.AwsClustersGrpcTransport, transports.AwsClustersRestTransport]
    assert transport in available_transports
    transport = AwsClustersClient.get_transport_class('grpc')
    assert transport == transports.AwsClustersGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AwsClustersClient, transports.AwsClustersGrpcTransport, 'grpc'), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio'), (AwsClustersClient, transports.AwsClustersRestTransport, 'rest')])
@mock.patch.object(AwsClustersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AwsClustersClient))
@mock.patch.object(AwsClustersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AwsClustersAsyncClient))
def test_aws_clusters_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(AwsClustersClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AwsClustersClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AwsClustersClient, transports.AwsClustersGrpcTransport, 'grpc', 'true'), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AwsClustersClient, transports.AwsClustersGrpcTransport, 'grpc', 'false'), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AwsClustersClient, transports.AwsClustersRestTransport, 'rest', 'true'), (AwsClustersClient, transports.AwsClustersRestTransport, 'rest', 'false')])
@mock.patch.object(AwsClustersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AwsClustersClient))
@mock.patch.object(AwsClustersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AwsClustersAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_aws_clusters_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AwsClustersClient, AwsClustersAsyncClient])
@mock.patch.object(AwsClustersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AwsClustersClient))
@mock.patch.object(AwsClustersAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AwsClustersAsyncClient))
def test_aws_clusters_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AwsClustersClient, transports.AwsClustersGrpcTransport, 'grpc'), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio'), (AwsClustersClient, transports.AwsClustersRestTransport, 'rest')])
def test_aws_clusters_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AwsClustersClient, transports.AwsClustersGrpcTransport, 'grpc', grpc_helpers), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AwsClustersClient, transports.AwsClustersRestTransport, 'rest', None)])
def test_aws_clusters_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_aws_clusters_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.gke_multicloud_v1.services.aws_clusters.transports.AwsClustersGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AwsClustersClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AwsClustersClient, transports.AwsClustersGrpcTransport, 'grpc', grpc_helpers), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_aws_clusters_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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

@pytest.mark.parametrize('request_type', [aws_service.CreateAwsClusterRequest, dict])
def test_create_aws_cluster(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.CreateAwsClusterRequest()
    assert isinstance(response, future.Future)

def test_create_aws_cluster_empty_call():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        client.create_aws_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.CreateAwsClusterRequest()

@pytest.mark.asyncio
async def test_create_aws_cluster_async(transport: str='grpc_asyncio', request_type=aws_service.CreateAwsClusterRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.CreateAwsClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_aws_cluster_async_from_dict():
    await test_create_aws_cluster_async(request_type=dict)

def test_create_aws_cluster_field_headers():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.CreateAwsClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_aws_cluster_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.CreateAwsClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_aws_cluster_flattened():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_aws_cluster(parent='parent_value', aws_cluster=aws_resources.AwsCluster(name='name_value'), aws_cluster_id='aws_cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].aws_cluster
        mock_val = aws_resources.AwsCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].aws_cluster_id
        mock_val = 'aws_cluster_id_value'
        assert arg == mock_val

def test_create_aws_cluster_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_aws_cluster(aws_service.CreateAwsClusterRequest(), parent='parent_value', aws_cluster=aws_resources.AwsCluster(name='name_value'), aws_cluster_id='aws_cluster_id_value')

@pytest.mark.asyncio
async def test_create_aws_cluster_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_aws_cluster(parent='parent_value', aws_cluster=aws_resources.AwsCluster(name='name_value'), aws_cluster_id='aws_cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].aws_cluster
        mock_val = aws_resources.AwsCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].aws_cluster_id
        mock_val = 'aws_cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_aws_cluster_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_aws_cluster(aws_service.CreateAwsClusterRequest(), parent='parent_value', aws_cluster=aws_resources.AwsCluster(name='name_value'), aws_cluster_id='aws_cluster_id_value')

@pytest.mark.parametrize('request_type', [aws_service.UpdateAwsClusterRequest, dict])
def test_update_aws_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.UpdateAwsClusterRequest()
    assert isinstance(response, future.Future)

def test_update_aws_cluster_empty_call():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        client.update_aws_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.UpdateAwsClusterRequest()

@pytest.mark.asyncio
async def test_update_aws_cluster_async(transport: str='grpc_asyncio', request_type=aws_service.UpdateAwsClusterRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.UpdateAwsClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_aws_cluster_async_from_dict():
    await test_update_aws_cluster_async(request_type=dict)

def test_update_aws_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.UpdateAwsClusterRequest()
    request.aws_cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'aws_cluster.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_aws_cluster_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.UpdateAwsClusterRequest()
    request.aws_cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'aws_cluster.name=name_value') in kw['metadata']

def test_update_aws_cluster_flattened():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_aws_cluster(aws_cluster=aws_resources.AwsCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].aws_cluster
        mock_val = aws_resources.AwsCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_aws_cluster_flattened_error():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_aws_cluster(aws_service.UpdateAwsClusterRequest(), aws_cluster=aws_resources.AwsCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_aws_cluster_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_aws_cluster(aws_cluster=aws_resources.AwsCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].aws_cluster
        mock_val = aws_resources.AwsCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_aws_cluster_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_aws_cluster(aws_service.UpdateAwsClusterRequest(), aws_cluster=aws_resources.AwsCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [aws_service.GetAwsClusterRequest, dict])
def test_get_aws_cluster(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        call.return_value = aws_resources.AwsCluster(name='name_value', description='description_value', aws_region='aws_region_value', state=aws_resources.AwsCluster.State.PROVISIONING, endpoint='endpoint_value', uid='uid_value', reconciling=True, etag='etag_value', cluster_ca_certificate='cluster_ca_certificate_value')
        response = client.get_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsClusterRequest()
    assert isinstance(response, aws_resources.AwsCluster)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.aws_region == 'aws_region_value'
    assert response.state == aws_resources.AwsCluster.State.PROVISIONING
    assert response.endpoint == 'endpoint_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.cluster_ca_certificate == 'cluster_ca_certificate_value'

def test_get_aws_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        client.get_aws_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsClusterRequest()

@pytest.mark.asyncio
async def test_get_aws_cluster_async(transport: str='grpc_asyncio', request_type=aws_service.GetAwsClusterRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsCluster(name='name_value', description='description_value', aws_region='aws_region_value', state=aws_resources.AwsCluster.State.PROVISIONING, endpoint='endpoint_value', uid='uid_value', reconciling=True, etag='etag_value', cluster_ca_certificate='cluster_ca_certificate_value'))
        response = await client.get_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsClusterRequest()
    assert isinstance(response, aws_resources.AwsCluster)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.aws_region == 'aws_region_value'
    assert response.state == aws_resources.AwsCluster.State.PROVISIONING
    assert response.endpoint == 'endpoint_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.cluster_ca_certificate == 'cluster_ca_certificate_value'

@pytest.mark.asyncio
async def test_get_aws_cluster_async_from_dict():
    await test_get_aws_cluster_async(request_type=dict)

def test_get_aws_cluster_field_headers():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GetAwsClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        call.return_value = aws_resources.AwsCluster()
        client.get_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_aws_cluster_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GetAwsClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsCluster())
        await client.get_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_aws_cluster_flattened():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        call.return_value = aws_resources.AwsCluster()
        client.get_aws_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_aws_cluster_flattened_error():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_aws_cluster(aws_service.GetAwsClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_aws_cluster_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_aws_cluster), '__call__') as call:
        call.return_value = aws_resources.AwsCluster()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsCluster())
        response = await client.get_aws_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_aws_cluster_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_aws_cluster(aws_service.GetAwsClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [aws_service.ListAwsClustersRequest, dict])
def test_list_aws_clusters(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.return_value = aws_service.ListAwsClustersResponse(next_page_token='next_page_token_value')
        response = client.list_aws_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.ListAwsClustersRequest()
    assert isinstance(response, pagers.ListAwsClustersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_aws_clusters_empty_call():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        client.list_aws_clusters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.ListAwsClustersRequest()

@pytest.mark.asyncio
async def test_list_aws_clusters_async(transport: str='grpc_asyncio', request_type=aws_service.ListAwsClustersRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.ListAwsClustersResponse(next_page_token='next_page_token_value'))
        response = await client.list_aws_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.ListAwsClustersRequest()
    assert isinstance(response, pagers.ListAwsClustersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_aws_clusters_async_from_dict():
    await test_list_aws_clusters_async(request_type=dict)

def test_list_aws_clusters_field_headers():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.ListAwsClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.return_value = aws_service.ListAwsClustersResponse()
        client.list_aws_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_aws_clusters_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.ListAwsClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.ListAwsClustersResponse())
        await client.list_aws_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_aws_clusters_flattened():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.return_value = aws_service.ListAwsClustersResponse()
        client.list_aws_clusters(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_aws_clusters_flattened_error():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_aws_clusters(aws_service.ListAwsClustersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_aws_clusters_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.return_value = aws_service.ListAwsClustersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.ListAwsClustersResponse())
        response = await client.list_aws_clusters(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_aws_clusters_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_aws_clusters(aws_service.ListAwsClustersRequest(), parent='parent_value')

def test_list_aws_clusters_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.side_effect = (aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster(), aws_resources.AwsCluster()], next_page_token='abc'), aws_service.ListAwsClustersResponse(aws_clusters=[], next_page_token='def'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster()], next_page_token='ghi'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_aws_clusters(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, aws_resources.AwsCluster) for i in results))

def test_list_aws_clusters_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__') as call:
        call.side_effect = (aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster(), aws_resources.AwsCluster()], next_page_token='abc'), aws_service.ListAwsClustersResponse(aws_clusters=[], next_page_token='def'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster()], next_page_token='ghi'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster()]), RuntimeError)
        pages = list(client.list_aws_clusters(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_aws_clusters_async_pager():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster(), aws_resources.AwsCluster()], next_page_token='abc'), aws_service.ListAwsClustersResponse(aws_clusters=[], next_page_token='def'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster()], next_page_token='ghi'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster()]), RuntimeError)
        async_pager = await client.list_aws_clusters(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, aws_resources.AwsCluster) for i in responses))

@pytest.mark.asyncio
async def test_list_aws_clusters_async_pages():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_aws_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster(), aws_resources.AwsCluster()], next_page_token='abc'), aws_service.ListAwsClustersResponse(aws_clusters=[], next_page_token='def'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster()], next_page_token='ghi'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_aws_clusters(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [aws_service.DeleteAwsClusterRequest, dict])
def test_delete_aws_cluster(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.DeleteAwsClusterRequest()
    assert isinstance(response, future.Future)

def test_delete_aws_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        client.delete_aws_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.DeleteAwsClusterRequest()

@pytest.mark.asyncio
async def test_delete_aws_cluster_async(transport: str='grpc_asyncio', request_type=aws_service.DeleteAwsClusterRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.DeleteAwsClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_aws_cluster_async_from_dict():
    await test_delete_aws_cluster_async(request_type=dict)

def test_delete_aws_cluster_field_headers():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.DeleteAwsClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_aws_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_aws_cluster_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.DeleteAwsClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_aws_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_aws_cluster_flattened():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_aws_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_aws_cluster_flattened_error():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_aws_cluster(aws_service.DeleteAwsClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_aws_cluster_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_aws_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_aws_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_aws_cluster_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_aws_cluster(aws_service.DeleteAwsClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [aws_service.GenerateAwsAccessTokenRequest, dict])
def test_generate_aws_access_token(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_aws_access_token), '__call__') as call:
        call.return_value = aws_service.GenerateAwsAccessTokenResponse(access_token='access_token_value')
        response = client.generate_aws_access_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GenerateAwsAccessTokenRequest()
    assert isinstance(response, aws_service.GenerateAwsAccessTokenResponse)
    assert response.access_token == 'access_token_value'

def test_generate_aws_access_token_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_aws_access_token), '__call__') as call:
        client.generate_aws_access_token()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GenerateAwsAccessTokenRequest()

@pytest.mark.asyncio
async def test_generate_aws_access_token_async(transport: str='grpc_asyncio', request_type=aws_service.GenerateAwsAccessTokenRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_aws_access_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.GenerateAwsAccessTokenResponse(access_token='access_token_value'))
        response = await client.generate_aws_access_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GenerateAwsAccessTokenRequest()
    assert isinstance(response, aws_service.GenerateAwsAccessTokenResponse)
    assert response.access_token == 'access_token_value'

@pytest.mark.asyncio
async def test_generate_aws_access_token_async_from_dict():
    await test_generate_aws_access_token_async(request_type=dict)

def test_generate_aws_access_token_field_headers():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GenerateAwsAccessTokenRequest()
    request.aws_cluster = 'aws_cluster_value'
    with mock.patch.object(type(client.transport.generate_aws_access_token), '__call__') as call:
        call.return_value = aws_service.GenerateAwsAccessTokenResponse()
        client.generate_aws_access_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'aws_cluster=aws_cluster_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_aws_access_token_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GenerateAwsAccessTokenRequest()
    request.aws_cluster = 'aws_cluster_value'
    with mock.patch.object(type(client.transport.generate_aws_access_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.GenerateAwsAccessTokenResponse())
        await client.generate_aws_access_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'aws_cluster=aws_cluster_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [aws_service.CreateAwsNodePoolRequest, dict])
def test_create_aws_node_pool(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.CreateAwsNodePoolRequest()
    assert isinstance(response, future.Future)

def test_create_aws_node_pool_empty_call():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        client.create_aws_node_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.CreateAwsNodePoolRequest()

@pytest.mark.asyncio
async def test_create_aws_node_pool_async(transport: str='grpc_asyncio', request_type=aws_service.CreateAwsNodePoolRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.CreateAwsNodePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_aws_node_pool_async_from_dict():
    await test_create_aws_node_pool_async(request_type=dict)

def test_create_aws_node_pool_field_headers():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.CreateAwsNodePoolRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_aws_node_pool_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.CreateAwsNodePoolRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_aws_node_pool_flattened():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_aws_node_pool(parent='parent_value', aws_node_pool=aws_resources.AwsNodePool(name='name_value'), aws_node_pool_id='aws_node_pool_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].aws_node_pool
        mock_val = aws_resources.AwsNodePool(name='name_value')
        assert arg == mock_val
        arg = args[0].aws_node_pool_id
        mock_val = 'aws_node_pool_id_value'
        assert arg == mock_val

def test_create_aws_node_pool_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_aws_node_pool(aws_service.CreateAwsNodePoolRequest(), parent='parent_value', aws_node_pool=aws_resources.AwsNodePool(name='name_value'), aws_node_pool_id='aws_node_pool_id_value')

@pytest.mark.asyncio
async def test_create_aws_node_pool_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_aws_node_pool(parent='parent_value', aws_node_pool=aws_resources.AwsNodePool(name='name_value'), aws_node_pool_id='aws_node_pool_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].aws_node_pool
        mock_val = aws_resources.AwsNodePool(name='name_value')
        assert arg == mock_val
        arg = args[0].aws_node_pool_id
        mock_val = 'aws_node_pool_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_aws_node_pool_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_aws_node_pool(aws_service.CreateAwsNodePoolRequest(), parent='parent_value', aws_node_pool=aws_resources.AwsNodePool(name='name_value'), aws_node_pool_id='aws_node_pool_id_value')

@pytest.mark.parametrize('request_type', [aws_service.UpdateAwsNodePoolRequest, dict])
def test_update_aws_node_pool(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.UpdateAwsNodePoolRequest()
    assert isinstance(response, future.Future)

def test_update_aws_node_pool_empty_call():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        client.update_aws_node_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.UpdateAwsNodePoolRequest()

@pytest.mark.asyncio
async def test_update_aws_node_pool_async(transport: str='grpc_asyncio', request_type=aws_service.UpdateAwsNodePoolRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.UpdateAwsNodePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_aws_node_pool_async_from_dict():
    await test_update_aws_node_pool_async(request_type=dict)

def test_update_aws_node_pool_field_headers():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.UpdateAwsNodePoolRequest()
    request.aws_node_pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'aws_node_pool.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_aws_node_pool_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.UpdateAwsNodePoolRequest()
    request.aws_node_pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'aws_node_pool.name=name_value') in kw['metadata']

def test_update_aws_node_pool_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_aws_node_pool(aws_node_pool=aws_resources.AwsNodePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].aws_node_pool
        mock_val = aws_resources.AwsNodePool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_aws_node_pool_flattened_error():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_aws_node_pool(aws_service.UpdateAwsNodePoolRequest(), aws_node_pool=aws_resources.AwsNodePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_aws_node_pool_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_aws_node_pool(aws_node_pool=aws_resources.AwsNodePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].aws_node_pool
        mock_val = aws_resources.AwsNodePool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_aws_node_pool_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_aws_node_pool(aws_service.UpdateAwsNodePoolRequest(), aws_node_pool=aws_resources.AwsNodePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [aws_service.GetAwsNodePoolRequest, dict])
def test_get_aws_node_pool(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        call.return_value = aws_resources.AwsNodePool(name='name_value', version='version_value', subnet_id='subnet_id_value', state=aws_resources.AwsNodePool.State.PROVISIONING, uid='uid_value', reconciling=True, etag='etag_value')
        response = client.get_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsNodePoolRequest()
    assert isinstance(response, aws_resources.AwsNodePool)
    assert response.name == 'name_value'
    assert response.version == 'version_value'
    assert response.subnet_id == 'subnet_id_value'
    assert response.state == aws_resources.AwsNodePool.State.PROVISIONING
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'

def test_get_aws_node_pool_empty_call():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        client.get_aws_node_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsNodePoolRequest()

@pytest.mark.asyncio
async def test_get_aws_node_pool_async(transport: str='grpc_asyncio', request_type=aws_service.GetAwsNodePoolRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsNodePool(name='name_value', version='version_value', subnet_id='subnet_id_value', state=aws_resources.AwsNodePool.State.PROVISIONING, uid='uid_value', reconciling=True, etag='etag_value'))
        response = await client.get_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsNodePoolRequest()
    assert isinstance(response, aws_resources.AwsNodePool)
    assert response.name == 'name_value'
    assert response.version == 'version_value'
    assert response.subnet_id == 'subnet_id_value'
    assert response.state == aws_resources.AwsNodePool.State.PROVISIONING
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_aws_node_pool_async_from_dict():
    await test_get_aws_node_pool_async(request_type=dict)

def test_get_aws_node_pool_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GetAwsNodePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        call.return_value = aws_resources.AwsNodePool()
        client.get_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_aws_node_pool_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GetAwsNodePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsNodePool())
        await client.get_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_aws_node_pool_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        call.return_value = aws_resources.AwsNodePool()
        client.get_aws_node_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_aws_node_pool_flattened_error():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_aws_node_pool(aws_service.GetAwsNodePoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_aws_node_pool_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_aws_node_pool), '__call__') as call:
        call.return_value = aws_resources.AwsNodePool()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsNodePool())
        response = await client.get_aws_node_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_aws_node_pool_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_aws_node_pool(aws_service.GetAwsNodePoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [aws_service.ListAwsNodePoolsRequest, dict])
def test_list_aws_node_pools(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.return_value = aws_service.ListAwsNodePoolsResponse(next_page_token='next_page_token_value')
        response = client.list_aws_node_pools(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.ListAwsNodePoolsRequest()
    assert isinstance(response, pagers.ListAwsNodePoolsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_aws_node_pools_empty_call():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        client.list_aws_node_pools()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.ListAwsNodePoolsRequest()

@pytest.mark.asyncio
async def test_list_aws_node_pools_async(transport: str='grpc_asyncio', request_type=aws_service.ListAwsNodePoolsRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.ListAwsNodePoolsResponse(next_page_token='next_page_token_value'))
        response = await client.list_aws_node_pools(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.ListAwsNodePoolsRequest()
    assert isinstance(response, pagers.ListAwsNodePoolsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_aws_node_pools_async_from_dict():
    await test_list_aws_node_pools_async(request_type=dict)

def test_list_aws_node_pools_field_headers():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.ListAwsNodePoolsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.return_value = aws_service.ListAwsNodePoolsResponse()
        client.list_aws_node_pools(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_aws_node_pools_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.ListAwsNodePoolsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.ListAwsNodePoolsResponse())
        await client.list_aws_node_pools(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_aws_node_pools_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.return_value = aws_service.ListAwsNodePoolsResponse()
        client.list_aws_node_pools(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_aws_node_pools_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_aws_node_pools(aws_service.ListAwsNodePoolsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_aws_node_pools_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.return_value = aws_service.ListAwsNodePoolsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_service.ListAwsNodePoolsResponse())
        response = await client.list_aws_node_pools(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_aws_node_pools_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_aws_node_pools(aws_service.ListAwsNodePoolsRequest(), parent='parent_value')

def test_list_aws_node_pools_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.side_effect = (aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool(), aws_resources.AwsNodePool()], next_page_token='abc'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[], next_page_token='def'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool()], next_page_token='ghi'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_aws_node_pools(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, aws_resources.AwsNodePool) for i in results))

def test_list_aws_node_pools_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__') as call:
        call.side_effect = (aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool(), aws_resources.AwsNodePool()], next_page_token='abc'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[], next_page_token='def'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool()], next_page_token='ghi'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool()]), RuntimeError)
        pages = list(client.list_aws_node_pools(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_aws_node_pools_async_pager():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool(), aws_resources.AwsNodePool()], next_page_token='abc'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[], next_page_token='def'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool()], next_page_token='ghi'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool()]), RuntimeError)
        async_pager = await client.list_aws_node_pools(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, aws_resources.AwsNodePool) for i in responses))

@pytest.mark.asyncio
async def test_list_aws_node_pools_async_pages():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_aws_node_pools), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool(), aws_resources.AwsNodePool()], next_page_token='abc'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[], next_page_token='def'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool()], next_page_token='ghi'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_aws_node_pools(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [aws_service.DeleteAwsNodePoolRequest, dict])
def test_delete_aws_node_pool(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.DeleteAwsNodePoolRequest()
    assert isinstance(response, future.Future)

def test_delete_aws_node_pool_empty_call():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        client.delete_aws_node_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.DeleteAwsNodePoolRequest()

@pytest.mark.asyncio
async def test_delete_aws_node_pool_async(transport: str='grpc_asyncio', request_type=aws_service.DeleteAwsNodePoolRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.DeleteAwsNodePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_aws_node_pool_async_from_dict():
    await test_delete_aws_node_pool_async(request_type=dict)

def test_delete_aws_node_pool_field_headers():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.DeleteAwsNodePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_aws_node_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_aws_node_pool_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.DeleteAwsNodePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_aws_node_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_aws_node_pool_flattened():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_aws_node_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_aws_node_pool_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_aws_node_pool(aws_service.DeleteAwsNodePoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_aws_node_pool_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_aws_node_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_aws_node_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_aws_node_pool_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_aws_node_pool(aws_service.DeleteAwsNodePoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [aws_service.GetAwsServerConfigRequest, dict])
def test_get_aws_server_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        call.return_value = aws_resources.AwsServerConfig(name='name_value', supported_aws_regions=['supported_aws_regions_value'])
        response = client.get_aws_server_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsServerConfigRequest()
    assert isinstance(response, aws_resources.AwsServerConfig)
    assert response.name == 'name_value'
    assert response.supported_aws_regions == ['supported_aws_regions_value']

def test_get_aws_server_config_empty_call():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        client.get_aws_server_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsServerConfigRequest()

@pytest.mark.asyncio
async def test_get_aws_server_config_async(transport: str='grpc_asyncio', request_type=aws_service.GetAwsServerConfigRequest):
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsServerConfig(name='name_value', supported_aws_regions=['supported_aws_regions_value']))
        response = await client.get_aws_server_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == aws_service.GetAwsServerConfigRequest()
    assert isinstance(response, aws_resources.AwsServerConfig)
    assert response.name == 'name_value'
    assert response.supported_aws_regions == ['supported_aws_regions_value']

@pytest.mark.asyncio
async def test_get_aws_server_config_async_from_dict():
    await test_get_aws_server_config_async(request_type=dict)

def test_get_aws_server_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GetAwsServerConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        call.return_value = aws_resources.AwsServerConfig()
        client.get_aws_server_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_aws_server_config_field_headers_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = aws_service.GetAwsServerConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsServerConfig())
        await client.get_aws_server_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_aws_server_config_flattened():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        call.return_value = aws_resources.AwsServerConfig()
        client.get_aws_server_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_aws_server_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_aws_server_config(aws_service.GetAwsServerConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_aws_server_config_flattened_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_aws_server_config), '__call__') as call:
        call.return_value = aws_resources.AwsServerConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(aws_resources.AwsServerConfig())
        response = await client.get_aws_server_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_aws_server_config_flattened_error_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_aws_server_config(aws_service.GetAwsServerConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [aws_service.CreateAwsClusterRequest, dict])
def test_create_aws_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['aws_cluster'] = {'name': 'name_value', 'description': 'description_value', 'networking': {'vpc_id': 'vpc_id_value', 'pod_address_cidr_blocks': ['pod_address_cidr_blocks_value1', 'pod_address_cidr_blocks_value2'], 'service_address_cidr_blocks': ['service_address_cidr_blocks_value1', 'service_address_cidr_blocks_value2']}, 'aws_region': 'aws_region_value', 'control_plane': {'version': 'version_value', 'instance_type': 'instance_type_value', 'ssh_config': {'ec2_key_pair': 'ec2_key_pair_value'}, 'subnet_ids': ['subnet_ids_value1', 'subnet_ids_value2'], 'security_group_ids': ['security_group_ids_value1', 'security_group_ids_value2'], 'iam_instance_profile': 'iam_instance_profile_value', 'root_volume': {'size_gib': 844, 'volume_type': 1, 'iops': 443, 'kms_key_arn': 'kms_key_arn_value'}, 'main_volume': {}, 'database_encryption': {'kms_key_arn': 'kms_key_arn_value'}, 'tags': {}, 'aws_services_authentication': {'role_arn': 'role_arn_value', 'role_session_name': 'role_session_name_value'}, 'proxy_config': {'secret_arn': 'secret_arn_value', 'secret_version': 'secret_version_value'}, 'config_encryption': {'kms_key_arn': 'kms_key_arn_value'}, 'instance_placement': {'tenancy': 1}}, 'authorization': {'admin_users': [{'username': 'username_value'}]}, 'state': 1, 'endpoint': 'endpoint_value', 'uid': 'uid_value', 'reconciling': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'annotations': {}, 'workload_identity_config': {'issuer_uri': 'issuer_uri_value', 'workload_pool': 'workload_pool_value', 'identity_provider': 'identity_provider_value'}, 'cluster_ca_certificate': 'cluster_ca_certificate_value', 'fleet': {'project': 'project_value', 'membership': 'membership_value'}, 'logging_config': {'component_config': {'enable_components': [1]}}, 'errors': [{'message': 'message_value'}], 'monitoring_config': {'managed_prometheus_config': {'enabled': True}}}
    test_field = aws_service.CreateAwsClusterRequest.meta.fields['aws_cluster']

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
    for (field, value) in request_init['aws_cluster'].items():
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
                for i in range(0, len(request_init['aws_cluster'][field])):
                    del request_init['aws_cluster'][field][i][subfield]
            else:
                del request_init['aws_cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_aws_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_create_aws_cluster_rest_required_fields(request_type=aws_service.CreateAwsClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['aws_cluster_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'awsClusterId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_aws_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'awsClusterId' in jsonified_request
    assert jsonified_request['awsClusterId'] == request_init['aws_cluster_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['awsClusterId'] = 'aws_cluster_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_aws_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('aws_cluster_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'awsClusterId' in jsonified_request
    assert jsonified_request['awsClusterId'] == 'aws_cluster_id_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_aws_cluster(request)
            expected_params = [('awsClusterId', '')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_aws_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_aws_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('awsClusterId', 'validateOnly')) & set(('parent', 'awsCluster', 'awsClusterId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_aws_cluster_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AwsClustersRestInterceptor, 'post_create_aws_cluster') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_create_aws_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.CreateAwsClusterRequest.pb(aws_service.CreateAwsClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = aws_service.CreateAwsClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_aws_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_aws_cluster_rest_bad_request(transport: str='rest', request_type=aws_service.CreateAwsClusterRequest):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_aws_cluster(request)

def test_create_aws_cluster_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', aws_cluster=aws_resources.AwsCluster(name='name_value'), aws_cluster_id='aws_cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_aws_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/awsClusters' % client.transport._host, args[1])

def test_create_aws_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_aws_cluster(aws_service.CreateAwsClusterRequest(), parent='parent_value', aws_cluster=aws_resources.AwsCluster(name='name_value'), aws_cluster_id='aws_cluster_id_value')

def test_create_aws_cluster_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.UpdateAwsClusterRequest, dict])
def test_update_aws_cluster_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'aws_cluster': {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}}
    request_init['aws_cluster'] = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3', 'description': 'description_value', 'networking': {'vpc_id': 'vpc_id_value', 'pod_address_cidr_blocks': ['pod_address_cidr_blocks_value1', 'pod_address_cidr_blocks_value2'], 'service_address_cidr_blocks': ['service_address_cidr_blocks_value1', 'service_address_cidr_blocks_value2']}, 'aws_region': 'aws_region_value', 'control_plane': {'version': 'version_value', 'instance_type': 'instance_type_value', 'ssh_config': {'ec2_key_pair': 'ec2_key_pair_value'}, 'subnet_ids': ['subnet_ids_value1', 'subnet_ids_value2'], 'security_group_ids': ['security_group_ids_value1', 'security_group_ids_value2'], 'iam_instance_profile': 'iam_instance_profile_value', 'root_volume': {'size_gib': 844, 'volume_type': 1, 'iops': 443, 'kms_key_arn': 'kms_key_arn_value'}, 'main_volume': {}, 'database_encryption': {'kms_key_arn': 'kms_key_arn_value'}, 'tags': {}, 'aws_services_authentication': {'role_arn': 'role_arn_value', 'role_session_name': 'role_session_name_value'}, 'proxy_config': {'secret_arn': 'secret_arn_value', 'secret_version': 'secret_version_value'}, 'config_encryption': {'kms_key_arn': 'kms_key_arn_value'}, 'instance_placement': {'tenancy': 1}}, 'authorization': {'admin_users': [{'username': 'username_value'}]}, 'state': 1, 'endpoint': 'endpoint_value', 'uid': 'uid_value', 'reconciling': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'annotations': {}, 'workload_identity_config': {'issuer_uri': 'issuer_uri_value', 'workload_pool': 'workload_pool_value', 'identity_provider': 'identity_provider_value'}, 'cluster_ca_certificate': 'cluster_ca_certificate_value', 'fleet': {'project': 'project_value', 'membership': 'membership_value'}, 'logging_config': {'component_config': {'enable_components': [1]}}, 'errors': [{'message': 'message_value'}], 'monitoring_config': {'managed_prometheus_config': {'enabled': True}}}
    test_field = aws_service.UpdateAwsClusterRequest.meta.fields['aws_cluster']

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
    for (field, value) in request_init['aws_cluster'].items():
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
                for i in range(0, len(request_init['aws_cluster'][field])):
                    del request_init['aws_cluster'][field][i][subfield]
            else:
                del request_init['aws_cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_aws_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_update_aws_cluster_rest_required_fields(request_type=aws_service.UpdateAwsClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_aws_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_aws_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_aws_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_aws_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_aws_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask', 'validateOnly')) & set(('awsCluster', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_aws_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AwsClustersRestInterceptor, 'post_update_aws_cluster') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_update_aws_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.UpdateAwsClusterRequest.pb(aws_service.UpdateAwsClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = aws_service.UpdateAwsClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_aws_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_aws_cluster_rest_bad_request(transport: str='rest', request_type=aws_service.UpdateAwsClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'aws_cluster': {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_aws_cluster(request)

def test_update_aws_cluster_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'aws_cluster': {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}}
        mock_args = dict(aws_cluster=aws_resources.AwsCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_aws_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{aws_cluster.name=projects/*/locations/*/awsClusters/*}' % client.transport._host, args[1])

def test_update_aws_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_aws_cluster(aws_service.UpdateAwsClusterRequest(), aws_cluster=aws_resources.AwsCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_aws_cluster_rest_error():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.GetAwsClusterRequest, dict])
def test_get_aws_cluster_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_resources.AwsCluster(name='name_value', description='description_value', aws_region='aws_region_value', state=aws_resources.AwsCluster.State.PROVISIONING, endpoint='endpoint_value', uid='uid_value', reconciling=True, etag='etag_value', cluster_ca_certificate='cluster_ca_certificate_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_resources.AwsCluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_aws_cluster(request)
    assert isinstance(response, aws_resources.AwsCluster)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.aws_region == 'aws_region_value'
    assert response.state == aws_resources.AwsCluster.State.PROVISIONING
    assert response.endpoint == 'endpoint_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.cluster_ca_certificate == 'cluster_ca_certificate_value'

def test_get_aws_cluster_rest_required_fields(request_type=aws_service.GetAwsClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_aws_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_aws_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = aws_resources.AwsCluster()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = aws_resources.AwsCluster.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_aws_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_aws_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_aws_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_aws_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AwsClustersRestInterceptor, 'post_get_aws_cluster') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_get_aws_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.GetAwsClusterRequest.pb(aws_service.GetAwsClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = aws_resources.AwsCluster.to_json(aws_resources.AwsCluster())
        request = aws_service.GetAwsClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = aws_resources.AwsCluster()
        client.get_aws_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_aws_cluster_rest_bad_request(transport: str='rest', request_type=aws_service.GetAwsClusterRequest):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_aws_cluster(request)

def test_get_aws_cluster_rest_flattened():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_resources.AwsCluster()
        sample_request = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_resources.AwsCluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_aws_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/awsClusters/*}' % client.transport._host, args[1])

def test_get_aws_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_aws_cluster(aws_service.GetAwsClusterRequest(), name='name_value')

def test_get_aws_cluster_rest_error():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.ListAwsClustersRequest, dict])
def test_list_aws_clusters_rest(request_type):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_service.ListAwsClustersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_service.ListAwsClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_aws_clusters(request)
    assert isinstance(response, pagers.ListAwsClustersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_aws_clusters_rest_required_fields(request_type=aws_service.ListAwsClustersRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_aws_clusters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_aws_clusters._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = aws_service.ListAwsClustersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = aws_service.ListAwsClustersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_aws_clusters(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_aws_clusters_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_aws_clusters._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_aws_clusters_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AwsClustersRestInterceptor, 'post_list_aws_clusters') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_list_aws_clusters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.ListAwsClustersRequest.pb(aws_service.ListAwsClustersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = aws_service.ListAwsClustersResponse.to_json(aws_service.ListAwsClustersResponse())
        request = aws_service.ListAwsClustersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = aws_service.ListAwsClustersResponse()
        client.list_aws_clusters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_aws_clusters_rest_bad_request(transport: str='rest', request_type=aws_service.ListAwsClustersRequest):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_aws_clusters(request)

def test_list_aws_clusters_rest_flattened():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_service.ListAwsClustersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_service.ListAwsClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_aws_clusters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/awsClusters' % client.transport._host, args[1])

def test_list_aws_clusters_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_aws_clusters(aws_service.ListAwsClustersRequest(), parent='parent_value')

def test_list_aws_clusters_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster(), aws_resources.AwsCluster()], next_page_token='abc'), aws_service.ListAwsClustersResponse(aws_clusters=[], next_page_token='def'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster()], next_page_token='ghi'), aws_service.ListAwsClustersResponse(aws_clusters=[aws_resources.AwsCluster(), aws_resources.AwsCluster()]))
        response = response + response
        response = tuple((aws_service.ListAwsClustersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_aws_clusters(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, aws_resources.AwsCluster) for i in results))
        pages = list(client.list_aws_clusters(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [aws_service.DeleteAwsClusterRequest, dict])
def test_delete_aws_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_aws_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_delete_aws_cluster_rest_required_fields(request_type=aws_service.DeleteAwsClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_aws_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_aws_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_aws_cluster(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_aws_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_aws_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_aws_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AwsClustersRestInterceptor, 'post_delete_aws_cluster') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_delete_aws_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.DeleteAwsClusterRequest.pb(aws_service.DeleteAwsClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = aws_service.DeleteAwsClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_aws_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_aws_cluster_rest_bad_request(transport: str='rest', request_type=aws_service.DeleteAwsClusterRequest):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_aws_cluster(request)

def test_delete_aws_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_aws_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/awsClusters/*}' % client.transport._host, args[1])

def test_delete_aws_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_aws_cluster(aws_service.DeleteAwsClusterRequest(), name='name_value')

def test_delete_aws_cluster_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.GenerateAwsAccessTokenRequest, dict])
def test_generate_aws_access_token_rest(request_type):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'aws_cluster': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_service.GenerateAwsAccessTokenResponse(access_token='access_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_service.GenerateAwsAccessTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_aws_access_token(request)
    assert isinstance(response, aws_service.GenerateAwsAccessTokenResponse)
    assert response.access_token == 'access_token_value'

def test_generate_aws_access_token_rest_required_fields(request_type=aws_service.GenerateAwsAccessTokenRequest):
    if False:
        return 10
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['aws_cluster'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_aws_access_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['awsCluster'] = 'aws_cluster_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_aws_access_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'awsCluster' in jsonified_request
    assert jsonified_request['awsCluster'] == 'aws_cluster_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = aws_service.GenerateAwsAccessTokenResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = aws_service.GenerateAwsAccessTokenResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_aws_access_token(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_aws_access_token_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_aws_access_token._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('awsCluster',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_aws_access_token_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AwsClustersRestInterceptor, 'post_generate_aws_access_token') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_generate_aws_access_token') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.GenerateAwsAccessTokenRequest.pb(aws_service.GenerateAwsAccessTokenRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = aws_service.GenerateAwsAccessTokenResponse.to_json(aws_service.GenerateAwsAccessTokenResponse())
        request = aws_service.GenerateAwsAccessTokenRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = aws_service.GenerateAwsAccessTokenResponse()
        client.generate_aws_access_token(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_aws_access_token_rest_bad_request(transport: str='rest', request_type=aws_service.GenerateAwsAccessTokenRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'aws_cluster': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_aws_access_token(request)

def test_generate_aws_access_token_rest_error():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.CreateAwsNodePoolRequest, dict])
def test_create_aws_node_pool_rest(request_type):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request_init['aws_node_pool'] = {'name': 'name_value', 'version': 'version_value', 'config': {'instance_type': 'instance_type_value', 'root_volume': {'size_gib': 844, 'volume_type': 1, 'iops': 443, 'kms_key_arn': 'kms_key_arn_value'}, 'taints': [{'key': 'key_value', 'value': 'value_value', 'effect': 1}], 'labels': {}, 'tags': {}, 'iam_instance_profile': 'iam_instance_profile_value', 'image_type': 'image_type_value', 'ssh_config': {'ec2_key_pair': 'ec2_key_pair_value'}, 'security_group_ids': ['security_group_ids_value1', 'security_group_ids_value2'], 'proxy_config': {'secret_arn': 'secret_arn_value', 'secret_version': 'secret_version_value'}, 'config_encryption': {'kms_key_arn': 'kms_key_arn_value'}, 'instance_placement': {'tenancy': 1}, 'autoscaling_metrics_collection': {'granularity': 'granularity_value', 'metrics': ['metrics_value1', 'metrics_value2']}}, 'autoscaling': {'min_node_count': 1489, 'max_node_count': 1491}, 'subnet_id': 'subnet_id_value', 'state': 1, 'uid': 'uid_value', 'reconciling': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'annotations': {}, 'max_pods_constraint': {'max_pods_per_node': 1798}, 'errors': [{'message': 'message_value'}]}
    test_field = aws_service.CreateAwsNodePoolRequest.meta.fields['aws_node_pool']

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
    for (field, value) in request_init['aws_node_pool'].items():
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
                for i in range(0, len(request_init['aws_node_pool'][field])):
                    del request_init['aws_node_pool'][field][i][subfield]
            else:
                del request_init['aws_node_pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_aws_node_pool(request)
    assert response.operation.name == 'operations/spam'

def test_create_aws_node_pool_rest_required_fields(request_type=aws_service.CreateAwsNodePoolRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['aws_node_pool_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'awsNodePoolId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_aws_node_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'awsNodePoolId' in jsonified_request
    assert jsonified_request['awsNodePoolId'] == request_init['aws_node_pool_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['awsNodePoolId'] = 'aws_node_pool_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_aws_node_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('aws_node_pool_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'awsNodePoolId' in jsonified_request
    assert jsonified_request['awsNodePoolId'] == 'aws_node_pool_id_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_aws_node_pool(request)
            expected_params = [('awsNodePoolId', '')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_aws_node_pool_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_aws_node_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('awsNodePoolId', 'validateOnly')) & set(('parent', 'awsNodePool', 'awsNodePoolId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_aws_node_pool_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AwsClustersRestInterceptor, 'post_create_aws_node_pool') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_create_aws_node_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.CreateAwsNodePoolRequest.pb(aws_service.CreateAwsNodePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = aws_service.CreateAwsNodePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_aws_node_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_aws_node_pool_rest_bad_request(transport: str='rest', request_type=aws_service.CreateAwsNodePoolRequest):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_aws_node_pool(request)

def test_create_aws_node_pool_rest_flattened():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
        mock_args = dict(parent='parent_value', aws_node_pool=aws_resources.AwsNodePool(name='name_value'), aws_node_pool_id='aws_node_pool_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_aws_node_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/awsClusters/*}/awsNodePools' % client.transport._host, args[1])

def test_create_aws_node_pool_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_aws_node_pool(aws_service.CreateAwsNodePoolRequest(), parent='parent_value', aws_node_pool=aws_resources.AwsNodePool(name='name_value'), aws_node_pool_id='aws_node_pool_id_value')

def test_create_aws_node_pool_rest_error():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.UpdateAwsNodePoolRequest, dict])
def test_update_aws_node_pool_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'aws_node_pool': {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}}
    request_init['aws_node_pool'] = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4', 'version': 'version_value', 'config': {'instance_type': 'instance_type_value', 'root_volume': {'size_gib': 844, 'volume_type': 1, 'iops': 443, 'kms_key_arn': 'kms_key_arn_value'}, 'taints': [{'key': 'key_value', 'value': 'value_value', 'effect': 1}], 'labels': {}, 'tags': {}, 'iam_instance_profile': 'iam_instance_profile_value', 'image_type': 'image_type_value', 'ssh_config': {'ec2_key_pair': 'ec2_key_pair_value'}, 'security_group_ids': ['security_group_ids_value1', 'security_group_ids_value2'], 'proxy_config': {'secret_arn': 'secret_arn_value', 'secret_version': 'secret_version_value'}, 'config_encryption': {'kms_key_arn': 'kms_key_arn_value'}, 'instance_placement': {'tenancy': 1}, 'autoscaling_metrics_collection': {'granularity': 'granularity_value', 'metrics': ['metrics_value1', 'metrics_value2']}}, 'autoscaling': {'min_node_count': 1489, 'max_node_count': 1491}, 'subnet_id': 'subnet_id_value', 'state': 1, 'uid': 'uid_value', 'reconciling': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'annotations': {}, 'max_pods_constraint': {'max_pods_per_node': 1798}, 'errors': [{'message': 'message_value'}]}
    test_field = aws_service.UpdateAwsNodePoolRequest.meta.fields['aws_node_pool']

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
    for (field, value) in request_init['aws_node_pool'].items():
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
                for i in range(0, len(request_init['aws_node_pool'][field])):
                    del request_init['aws_node_pool'][field][i][subfield]
            else:
                del request_init['aws_node_pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_aws_node_pool(request)
    assert response.operation.name == 'operations/spam'

def test_update_aws_node_pool_rest_required_fields(request_type=aws_service.UpdateAwsNodePoolRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_aws_node_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_aws_node_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_aws_node_pool(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_aws_node_pool_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_aws_node_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask', 'validateOnly')) & set(('awsNodePool', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_aws_node_pool_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AwsClustersRestInterceptor, 'post_update_aws_node_pool') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_update_aws_node_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.UpdateAwsNodePoolRequest.pb(aws_service.UpdateAwsNodePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = aws_service.UpdateAwsNodePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_aws_node_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_aws_node_pool_rest_bad_request(transport: str='rest', request_type=aws_service.UpdateAwsNodePoolRequest):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'aws_node_pool': {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_aws_node_pool(request)

def test_update_aws_node_pool_rest_flattened():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'aws_node_pool': {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}}
        mock_args = dict(aws_node_pool=aws_resources.AwsNodePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_aws_node_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{aws_node_pool.name=projects/*/locations/*/awsClusters/*/awsNodePools/*}' % client.transport._host, args[1])

def test_update_aws_node_pool_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_aws_node_pool(aws_service.UpdateAwsNodePoolRequest(), aws_node_pool=aws_resources.AwsNodePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_aws_node_pool_rest_error():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.GetAwsNodePoolRequest, dict])
def test_get_aws_node_pool_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_resources.AwsNodePool(name='name_value', version='version_value', subnet_id='subnet_id_value', state=aws_resources.AwsNodePool.State.PROVISIONING, uid='uid_value', reconciling=True, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_resources.AwsNodePool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_aws_node_pool(request)
    assert isinstance(response, aws_resources.AwsNodePool)
    assert response.name == 'name_value'
    assert response.version == 'version_value'
    assert response.subnet_id == 'subnet_id_value'
    assert response.state == aws_resources.AwsNodePool.State.PROVISIONING
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'

def test_get_aws_node_pool_rest_required_fields(request_type=aws_service.GetAwsNodePoolRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_aws_node_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_aws_node_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = aws_resources.AwsNodePool()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = aws_resources.AwsNodePool.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_aws_node_pool(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_aws_node_pool_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_aws_node_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_aws_node_pool_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AwsClustersRestInterceptor, 'post_get_aws_node_pool') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_get_aws_node_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.GetAwsNodePoolRequest.pb(aws_service.GetAwsNodePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = aws_resources.AwsNodePool.to_json(aws_resources.AwsNodePool())
        request = aws_service.GetAwsNodePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = aws_resources.AwsNodePool()
        client.get_aws_node_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_aws_node_pool_rest_bad_request(transport: str='rest', request_type=aws_service.GetAwsNodePoolRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_aws_node_pool(request)

def test_get_aws_node_pool_rest_flattened():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_resources.AwsNodePool()
        sample_request = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_resources.AwsNodePool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_aws_node_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/awsClusters/*/awsNodePools/*}' % client.transport._host, args[1])

def test_get_aws_node_pool_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_aws_node_pool(aws_service.GetAwsNodePoolRequest(), name='name_value')

def test_get_aws_node_pool_rest_error():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.ListAwsNodePoolsRequest, dict])
def test_list_aws_node_pools_rest(request_type):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_service.ListAwsNodePoolsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_service.ListAwsNodePoolsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_aws_node_pools(request)
    assert isinstance(response, pagers.ListAwsNodePoolsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_aws_node_pools_rest_required_fields(request_type=aws_service.ListAwsNodePoolsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_aws_node_pools._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_aws_node_pools._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = aws_service.ListAwsNodePoolsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = aws_service.ListAwsNodePoolsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_aws_node_pools(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_aws_node_pools_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_aws_node_pools._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_aws_node_pools_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AwsClustersRestInterceptor, 'post_list_aws_node_pools') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_list_aws_node_pools') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.ListAwsNodePoolsRequest.pb(aws_service.ListAwsNodePoolsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = aws_service.ListAwsNodePoolsResponse.to_json(aws_service.ListAwsNodePoolsResponse())
        request = aws_service.ListAwsNodePoolsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = aws_service.ListAwsNodePoolsResponse()
        client.list_aws_node_pools(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_aws_node_pools_rest_bad_request(transport: str='rest', request_type=aws_service.ListAwsNodePoolsRequest):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_aws_node_pools(request)

def test_list_aws_node_pools_rest_flattened():
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_service.ListAwsNodePoolsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_service.ListAwsNodePoolsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_aws_node_pools(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/awsClusters/*}/awsNodePools' % client.transport._host, args[1])

def test_list_aws_node_pools_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_aws_node_pools(aws_service.ListAwsNodePoolsRequest(), parent='parent_value')

def test_list_aws_node_pools_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool(), aws_resources.AwsNodePool()], next_page_token='abc'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[], next_page_token='def'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool()], next_page_token='ghi'), aws_service.ListAwsNodePoolsResponse(aws_node_pools=[aws_resources.AwsNodePool(), aws_resources.AwsNodePool()]))
        response = response + response
        response = tuple((aws_service.ListAwsNodePoolsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/awsClusters/sample3'}
        pager = client.list_aws_node_pools(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, aws_resources.AwsNodePool) for i in results))
        pages = list(client.list_aws_node_pools(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [aws_service.DeleteAwsNodePoolRequest, dict])
def test_delete_aws_node_pool_rest(request_type):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_aws_node_pool(request)
    assert response.operation.name == 'operations/spam'

def test_delete_aws_node_pool_rest_required_fields(request_type=aws_service.DeleteAwsNodePoolRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_aws_node_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_aws_node_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_aws_node_pool(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_aws_node_pool_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_aws_node_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_aws_node_pool_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AwsClustersRestInterceptor, 'post_delete_aws_node_pool') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_delete_aws_node_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.DeleteAwsNodePoolRequest.pb(aws_service.DeleteAwsNodePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = aws_service.DeleteAwsNodePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_aws_node_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_aws_node_pool_rest_bad_request(transport: str='rest', request_type=aws_service.DeleteAwsNodePoolRequest):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_aws_node_pool(request)

def test_delete_aws_node_pool_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/awsClusters/sample3/awsNodePools/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_aws_node_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/awsClusters/*/awsNodePools/*}' % client.transport._host, args[1])

def test_delete_aws_node_pool_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_aws_node_pool(aws_service.DeleteAwsNodePoolRequest(), name='name_value')

def test_delete_aws_node_pool_rest_error():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [aws_service.GetAwsServerConfigRequest, dict])
def test_get_aws_server_config_rest(request_type):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/awsServerConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_resources.AwsServerConfig(name='name_value', supported_aws_regions=['supported_aws_regions_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_resources.AwsServerConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_aws_server_config(request)
    assert isinstance(response, aws_resources.AwsServerConfig)
    assert response.name == 'name_value'
    assert response.supported_aws_regions == ['supported_aws_regions_value']

def test_get_aws_server_config_rest_required_fields(request_type=aws_service.GetAwsServerConfigRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AwsClustersRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_aws_server_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_aws_server_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = aws_resources.AwsServerConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = aws_resources.AwsServerConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_aws_server_config(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_aws_server_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_aws_server_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_aws_server_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AwsClustersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AwsClustersRestInterceptor())
    client = AwsClustersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AwsClustersRestInterceptor, 'post_get_aws_server_config') as post, mock.patch.object(transports.AwsClustersRestInterceptor, 'pre_get_aws_server_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = aws_service.GetAwsServerConfigRequest.pb(aws_service.GetAwsServerConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = aws_resources.AwsServerConfig.to_json(aws_resources.AwsServerConfig())
        request = aws_service.GetAwsServerConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = aws_resources.AwsServerConfig()
        client.get_aws_server_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_aws_server_config_rest_bad_request(transport: str='rest', request_type=aws_service.GetAwsServerConfigRequest):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/awsServerConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_aws_server_config(request)

def test_get_aws_server_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = aws_resources.AwsServerConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/awsServerConfig'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = aws_resources.AwsServerConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_aws_server_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/awsServerConfig}' % client.transport._host, args[1])

def test_get_aws_server_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_aws_server_config(aws_service.GetAwsServerConfigRequest(), name='name_value')

def test_get_aws_server_config_rest_error():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.AwsClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AwsClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AwsClustersClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AwsClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AwsClustersClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AwsClustersClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AwsClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AwsClustersClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AwsClustersClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.AwsClustersGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AwsClustersGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AwsClustersGrpcTransport, transports.AwsClustersGrpcAsyncIOTransport, transports.AwsClustersRestTransport])
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
    transport = AwsClustersClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AwsClustersGrpcTransport)

def test_aws_clusters_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AwsClustersTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_aws_clusters_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.gke_multicloud_v1.services.aws_clusters.transports.AwsClustersTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AwsClustersTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_aws_cluster', 'update_aws_cluster', 'get_aws_cluster', 'list_aws_clusters', 'delete_aws_cluster', 'generate_aws_access_token', 'create_aws_node_pool', 'update_aws_node_pool', 'get_aws_node_pool', 'list_aws_node_pools', 'delete_aws_node_pool', 'get_aws_server_config', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_aws_clusters_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.gke_multicloud_v1.services.aws_clusters.transports.AwsClustersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AwsClustersTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_aws_clusters_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.gke_multicloud_v1.services.aws_clusters.transports.AwsClustersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AwsClustersTransport()
        adc.assert_called_once()

def test_aws_clusters_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AwsClustersClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AwsClustersGrpcTransport, transports.AwsClustersGrpcAsyncIOTransport])
def test_aws_clusters_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AwsClustersGrpcTransport, transports.AwsClustersGrpcAsyncIOTransport, transports.AwsClustersRestTransport])
def test_aws_clusters_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AwsClustersGrpcTransport, grpc_helpers), (transports.AwsClustersGrpcAsyncIOTransport, grpc_helpers_async)])
def test_aws_clusters_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('gkemulticloud.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='gkemulticloud.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AwsClustersGrpcTransport, transports.AwsClustersGrpcAsyncIOTransport])
def test_aws_clusters_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_aws_clusters_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AwsClustersRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_aws_clusters_rest_lro_client():
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_aws_clusters_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkemulticloud.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('gkemulticloud.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_aws_clusters_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkemulticloud.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('gkemulticloud.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkemulticloud.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_aws_clusters_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AwsClustersClient(credentials=creds1, transport=transport_name)
    client2 = AwsClustersClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_aws_cluster._session
    session2 = client2.transport.create_aws_cluster._session
    assert session1 != session2
    session1 = client1.transport.update_aws_cluster._session
    session2 = client2.transport.update_aws_cluster._session
    assert session1 != session2
    session1 = client1.transport.get_aws_cluster._session
    session2 = client2.transport.get_aws_cluster._session
    assert session1 != session2
    session1 = client1.transport.list_aws_clusters._session
    session2 = client2.transport.list_aws_clusters._session
    assert session1 != session2
    session1 = client1.transport.delete_aws_cluster._session
    session2 = client2.transport.delete_aws_cluster._session
    assert session1 != session2
    session1 = client1.transport.generate_aws_access_token._session
    session2 = client2.transport.generate_aws_access_token._session
    assert session1 != session2
    session1 = client1.transport.create_aws_node_pool._session
    session2 = client2.transport.create_aws_node_pool._session
    assert session1 != session2
    session1 = client1.transport.update_aws_node_pool._session
    session2 = client2.transport.update_aws_node_pool._session
    assert session1 != session2
    session1 = client1.transport.get_aws_node_pool._session
    session2 = client2.transport.get_aws_node_pool._session
    assert session1 != session2
    session1 = client1.transport.list_aws_node_pools._session
    session2 = client2.transport.list_aws_node_pools._session
    assert session1 != session2
    session1 = client1.transport.delete_aws_node_pool._session
    session2 = client2.transport.delete_aws_node_pool._session
    assert session1 != session2
    session1 = client1.transport.get_aws_server_config._session
    session2 = client2.transport.get_aws_server_config._session
    assert session1 != session2

def test_aws_clusters_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AwsClustersGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_aws_clusters_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AwsClustersGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AwsClustersGrpcTransport, transports.AwsClustersGrpcAsyncIOTransport])
def test_aws_clusters_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class', [transports.AwsClustersGrpcTransport, transports.AwsClustersGrpcAsyncIOTransport])
def test_aws_clusters_transport_channel_mtls_with_adc(transport_class):
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

def test_aws_clusters_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_aws_clusters_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_aws_cluster_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    aws_cluster = 'whelk'
    expected = 'projects/{project}/locations/{location}/awsClusters/{aws_cluster}'.format(project=project, location=location, aws_cluster=aws_cluster)
    actual = AwsClustersClient.aws_cluster_path(project, location, aws_cluster)
    assert expected == actual

def test_parse_aws_cluster_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'aws_cluster': 'nudibranch'}
    path = AwsClustersClient.aws_cluster_path(**expected)
    actual = AwsClustersClient.parse_aws_cluster_path(path)
    assert expected == actual

def test_aws_node_pool_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    aws_cluster = 'winkle'
    aws_node_pool = 'nautilus'
    expected = 'projects/{project}/locations/{location}/awsClusters/{aws_cluster}/awsNodePools/{aws_node_pool}'.format(project=project, location=location, aws_cluster=aws_cluster, aws_node_pool=aws_node_pool)
    actual = AwsClustersClient.aws_node_pool_path(project, location, aws_cluster, aws_node_pool)
    assert expected == actual

def test_parse_aws_node_pool_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone', 'aws_cluster': 'squid', 'aws_node_pool': 'clam'}
    path = AwsClustersClient.aws_node_pool_path(**expected)
    actual = AwsClustersClient.parse_aws_node_pool_path(path)
    assert expected == actual

def test_aws_server_config_path():
    if False:
        return 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}/awsServerConfig'.format(project=project, location=location)
    actual = AwsClustersClient.aws_server_config_path(project, location)
    assert expected == actual

def test_parse_aws_server_config_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = AwsClustersClient.aws_server_config_path(**expected)
    actual = AwsClustersClient.parse_aws_server_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AwsClustersClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'mussel'}
    path = AwsClustersClient.common_billing_account_path(**expected)
    actual = AwsClustersClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AwsClustersClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = AwsClustersClient.common_folder_path(**expected)
    actual = AwsClustersClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AwsClustersClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = AwsClustersClient.common_organization_path(**expected)
    actual = AwsClustersClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = AwsClustersClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'clam'}
    path = AwsClustersClient.common_project_path(**expected)
    actual = AwsClustersClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AwsClustersClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = AwsClustersClient.common_location_path(**expected)
    actual = AwsClustersClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AwsClustersTransport, '_prep_wrapped_messages') as prep:
        client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AwsClustersTransport, '_prep_wrapped_messages') as prep:
        transport_class = AwsClustersClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AwsClustersAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = AwsClustersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AwsClustersClient, transports.AwsClustersGrpcTransport), (AwsClustersAsyncClient, transports.AwsClustersGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)